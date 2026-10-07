// Matrix-multiplication kernels shared by the benchmark (sgemm.cu) and the
// Python shared library (python/lab6lib.cu). All matrices are N x N, row-major.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ---------------------------------------------------------------------------
// 1. Naive (handout): one thread per output element. threadIdx.x walks along
//    a row of C, so a warp reads 32 consecutive floats of B (coalesced) and
//    all 32 threads read the same float of A (a broadcast). Every operand
//    still comes from global memory: 2N loads per 2N flops.
__global__ void k_naive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// 2. Shared-memory tiling (handout, templated on the tile width). A T x T
//    tile of A and of B is loaded once into shared memory and each value is
//    then used T times, cutting global traffic by a factor of T.
template <int T>
__global__ void k_tiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int m = 0; m < (N + T - 1) / T; ++m) {
        int ac = m * T + tx, br = m * T + ty;
        As[ty][tx] = (row < N && ac < N) ? A[row * N + ac] : 0.0f;
        Bs[ty][tx] = (br < N && col < N) ? B[br * N + col] : 0.0f;
        __syncthreads();                    // tile fully loaded before anyone reads it
#pragma unroll
        for (int k = 0; k < T; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                    // everyone done before the tile is overwritten
    }
    if (row < N && col < N)
        C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3. Register blocking (extension). A block computes a BM x BN tile of C with
//    (BM*BN)/(TM*TN) threads; each thread computes a TM x TN sub-tile held in
//    registers. Per step of the inner loop a thread reads TM + TN values from
//    shared memory and does TM * TN multiply-adds, so shared-memory traffic
//    per flop drops by ~TM*TN/(TM+TN) = 4x for 8x8, and the tiled kernel's
//    real bottleneck (shared-memory bandwidth) moves to arithmetic.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
k_regblock(const float *A, const float *B, float *C, int N)
{
    constexpr int NT = (BM * BN) / (TM * TN);
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    constexpr int strideA = NT / BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
    constexpr int strideB = NT / BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];
    for (int bk = 0; bk < N; bk += BK) {
#pragma unroll
        for (int o = 0; o < BM; o += strideA)
            As[(innerRowA + o) * BK + innerColA] = A[(innerRowA + o) * N + innerColA];
#pragma unroll
        for (int o = 0; o < BK; o += strideB)
            Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int d = 0; d < BK; ++d) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
#pragma unroll
            for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + threadCol * TN + i];
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n)
            C[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
}

// ---------------------------------------------------------------------------
// 4. Register blocking + 128-bit memory operations (extension). Same tiling as
//    regblock, but global loads and stores move 4 floats per instruction
//    (float4), and the A tile is stored transposed in shared memory so the
//    inner loop reads it with unit stride as well.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
k_vec4(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[BK * BM];           // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // With 128x8 and 8x128 tiles and 256 threads, each thread moves exactly
    // one float4 of A and one float4 of B per k-step.
    const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];
    for (int bk = 0; bk < N; bk += BK) {
        float4 a = reinterpret_cast<const float4 *>(&A[innerRowA * N + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int d = 0; d < BK; ++d) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + threadRow * TM + i];
#pragma unroll
            for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + threadCol * TN + i];
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 v = make_float4(acc[m * TN + n], acc[m * TN + n + 1],
                                   acc[m * TN + n + 2], acc[m * TN + n + 3]);
            reinterpret_cast<float4 *>(&C[(threadRow * TM + m) * N + threadCol * TN + n])[0] = v;
        }
}

// ---------------------------------------------------------------------------
// 5. Tensor Cores via WMMA (extension; sm_70+, the T4 is sm_75). Inputs are
//    FP16, accumulation FP32. One warp computes one 16x16 tile of C straight
//    from global memory: deliberately simple, to show what the hardware unit
//    alone buys and what precision it costs.
using namespace nvcuda;
__global__ void k_wmma(const half *A, const half *B, float *C, int N)
{
    int tileRow = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tileCol = blockIdx.y * blockDim.y + threadIdx.y;
    int aRow = tileRow * 16, bCol = tileCol * 16;
    if (aRow >= N || bCol >= N) return;     // uniform across the warp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a, A + aRow * N + k, N);
        wmma::load_matrix_sync(b, B + k * N + bCol, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + aRow * N + bCol, c, N, wmma::mem_row_major);
}

__global__ void k_to_half(const float *in, half *out, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

