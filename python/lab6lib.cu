// Shared library exposing the CUDA kernels to Python through ctypes
// (handout Part 7 and Step 8.3).  Build: make lib  ->  bin/liblab6.so
//
// Two calling styles, on purpose:
//  - gpu_matrix_multiply: the handout's interface. Every call allocates device
//    memory, copies both inputs over PCIe, runs the tiled kernel, copies the
//    result back and frees everything.
//  - lab6_create / lab6_matmul / lab6_conv2d: a context that keeps device
//    buffers alive across calls and grows them only when a larger input
//    arrives, and records the kernel time separately so Python can tell
//    compute from overhead.
// Every function returns 0 on success or a cudaError_t / cublasStatus_t code.
#include <cstdio>
#include <cublas_v2.h>
#include "../gpu/kernels.cuh"
#include "../gpu/conv_kernels.cuh"

#define TRY(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "lab6: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return (int)e_; } } while (0)

extern "C" {

int gpu_matrix_multiply(const float *hA, const float *hB, float *hC, int N)
{
    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    TRY(cudaMalloc(&dA, bytes));
    TRY(cudaMalloc(&dB, bytes));
    TRY(cudaMalloc(&dC, bytes));
    TRY(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
    k_tiled<16><<<grd, blk>>>(dA, dB, dC, N);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    TRY(cudaFree(dA)); TRY(cudaFree(dB)); TRY(cudaFree(dC));
    return 0;
}

struct Lab6Ctx {
    size_t mm_cap = 0;                       // bytes per matrix buffer
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t px_cap = 0;                       // pixels in the image buffers
    unsigned char *dImg = nullptr;
    float *dOut = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t e0, e1;
    float kernel_ms = 0.0f;
};

void *lab6_create(void)
{
    Lab6Ctx *c = new Lab6Ctx;
    if (cudaFree(0) != cudaSuccess || cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS ||
        cudaEventCreate(&c->e0) != cudaSuccess || cudaEventCreate(&c->e1) != cudaSuccess) {
        delete c;
        return nullptr;
    }
    return c;
}

void lab6_destroy(void *p)
{
    Lab6Ctx *c = (Lab6Ctx *)p;
    if (!c) return;
    cudaFree(c->dA); cudaFree(c->dB); cudaFree(c->dC);
    cudaFree(c->dImg); cudaFree(c->dOut);
    cublasDestroy(c->blas);
    cudaEventDestroy(c->e0); cudaEventDestroy(c->e1);
    delete c;
}

float lab6_last_kernel_ms(void *p) { return ((Lab6Ctx *)p)->kernel_ms; }

// kernel: 0 = tiled16, 1 = regblock, 2 = vec4, 3 = cuBLAS. 1 and 2 need N % 128 == 0.
int lab6_matmul(void *p, const float *hA, const float *hB, float *hC, int N, int kernel)
{
    Lab6Ctx *c = (Lab6Ctx *)p;
    size_t bytes = (size_t)N * N * sizeof(float);
    if ((kernel == 1 || kernel == 2) && N % 128 != 0) return (int)cudaErrorInvalidValue;
    if (bytes > c->mm_cap) {                 // grow once, then reuse
        cudaFree(c->dA); cudaFree(c->dB); cudaFree(c->dC);
        TRY(cudaMalloc(&c->dA, bytes));
        TRY(cudaMalloc(&c->dB, bytes));
        TRY(cudaMalloc(&c->dC, bytes));
        c->mm_cap = bytes;
    }
    TRY(cudaMemcpy(c->dA, hA, bytes, cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(c->dB, hB, bytes, cudaMemcpyHostToDevice));
    TRY(cudaEventRecord(c->e0));
    if (kernel == 0) {
        dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
        k_tiled<16><<<grd, blk>>>(c->dA, c->dB, c->dC, N);
    } else if (kernel == 1) {
        k_regblock<128, 128, 8, 8, 8><<<dim3(N / 128, N / 128), 256>>>(c->dA, c->dB, c->dC, N);
    } else if (kernel == 2) {
        k_vec4<128, 128, 8, 8, 8><<<dim3(N / 128, N / 128), 256>>>(c->dA, c->dB, c->dC, N);
    } else if (kernel == 3) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t s = cublasSgemm(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                       &alpha, c->dB, N, c->dA, N, &beta, c->dC, N);
        if (s != CUBLAS_STATUS_SUCCESS) return 1000 + (int)s;
    } else {
        return (int)cudaErrorInvalidValue;
    }
    TRY(cudaGetLastError());
    TRY(cudaEventRecord(c->e1));
    TRY(cudaEventSynchronize(c->e1));
    TRY(cudaEventElapsedTime(&c->kernel_ms, c->e0, c->e1));
    TRY(cudaMemcpy(hC, c->dC, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// img: M x M uint8, filt: N x N float (N odd, <= 31), out: M x M float.
int lab6_conv2d(void *p, const unsigned char *img, float *out, int M, const float *filt, int N, int tiled)
{
    Lab6Ctx *c = (Lab6Ctx *)p;
    if (N <= 0 || N % 2 == 0 || N > CONV_MAX_N) return (int)cudaErrorInvalidValue;
    size_t px = (size_t)M * M;
    if (px > c->px_cap) {
        cudaFree(c->dImg); cudaFree(c->dOut);
        TRY(cudaMalloc(&c->dImg, px));
        TRY(cudaMalloc(&c->dOut, px * sizeof(float)));
        c->px_cap = px;
    }
    TRY(cudaMemcpy(c->dImg, img, px, cudaMemcpyHostToDevice));
    TRY(cudaMemcpyToSymbol(c_filter, filt, (size_t)N * N * sizeof(float)));
    TRY(cudaEventRecord(c->e0));
    conv_launch(tiled != 0, c->dImg, c->dOut, M, N);
    TRY(cudaGetLastError());
    TRY(cudaEventRecord(c->e1));
    TRY(cudaEventSynchronize(c->e1));
    TRY(cudaEventElapsedTime(&c->kernel_ms, c->e0, c->e1));
    TRY(cudaMemcpy(out, c->dOut, px * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

}  // extern "C"
