// 2-D convolution kernels shared by conv_gpu.cu and the Python library.
// Same semantics as conv/conv_cpu.c: 8-bit M x M image, N x N float filter
// (N odd, N <= CONV_MAX_N), float output, borders clamp to the edge.
#pragma once
#include <cuda_runtime.h>

#define CONV_MAX_N 31
#define CONV_TILE 16

// Every thread of a warp reads the same filter tap at the same time, which is
// exactly the access pattern the constant cache broadcasts in one cycle.
__constant__ float c_filter[CONV_MAX_N * CONV_MAX_N];

__device__ __forceinline__ int conv_clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// One thread per output pixel, image read straight from global memory. Each
// input pixel is fetched by up to N*N neighbouring threads.
__global__ void k_conv_naive(const unsigned char *img, float *out, int M, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= M) return;
    int r = N / 2;
    float acc = 0.0f;
    for (int i = 0; i < N; i++) {
        int yy = conv_clamp(y + i - r, 0, M - 1);
        for (int j = 0; j < N; j++)
            acc += img[yy * M + conv_clamp(x + j - r, 0, M - 1)] * c_filter[i * N + j];
    }
    out[y * M + x] = acc;
}

// A CONV_TILE x CONV_TILE block of outputs needs a (CONV_TILE + N - 1)^2
// input window: the tile plus an r-pixel halo on every side. The block loads
// that window into shared memory once, cooperatively, then every thread reads
// its N x N neighbourhood from shared memory.
__global__ void k_conv_tiled(const unsigned char *img, float *out, int M, int N)
{
    extern __shared__ unsigned char tile[];
    const int r = N / 2, W = CONV_TILE + N - 1;
    const int bx = blockIdx.x * CONV_TILE, by = blockIdx.y * CONV_TILE;
    for (int idx = threadIdx.y * CONV_TILE + threadIdx.x; idx < W * W; idx += CONV_TILE * CONV_TILE) {
        int ty = idx / W, tx = idx % W;
        tile[idx] = img[conv_clamp(by + ty - r, 0, M - 1) * M + conv_clamp(bx + tx - r, 0, M - 1)];
    }
    __syncthreads();
    int x = bx + threadIdx.x, y = by + threadIdx.y;
    if (x >= M || y >= M) return;
    float acc = 0.0f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            acc += tile[(threadIdx.y + i) * W + threadIdx.x + j] * c_filter[i * N + j];
    out[y * M + x] = acc;
}

static inline void conv_launch(bool tiled, const unsigned char *dImg, float *dOut, int M, int N)
{
    dim3 blk(CONV_TILE, CONV_TILE), grd((M + CONV_TILE - 1) / CONV_TILE, (M + CONV_TILE - 1) / CONV_TILE);
    if (tiled) {
        size_t shmem = (size_t)(CONV_TILE + N - 1) * (CONV_TILE + N - 1);
        k_conv_tiled<<<grd, blk, shmem>>>(dImg, dOut, M, N);
    } else {
        k_conv_naive<<<grd, blk>>>(dImg, dOut, M, N);
    }
}
