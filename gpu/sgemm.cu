// N x N single-precision matrix multiplication C = A * B (row-major) on an
// NVIDIA GPU, from the handout's naive kernel up to register-blocked tiles,
// Tensor Cores and cuBLAS.
//
//   ./sgemm <kernel> <N> [reps]
//   kernel: naive | tiled16 | tiled32 | regblock | vec4 | wmma | cublas | cublas_fp16 | all
//
// Prints CSV lines: impl,N,kernel_ms,gflops,e2e_ms,max_rel_err
//   kernel_ms  - mean over reps of the kernel alone (cudaEvent timing, after a warm-up)
//   e2e_ms     - one call including cudaMalloc, host->device copies, kernel and
//                device->host copy: what a caller that starts from host memory pays
//   max_rel_err- max |C - C_ref| / max|C_ref| against cuBLAS FP32
//
// regblock, vec4 and wmma need N to be a multiple of 128 (tiles are not padded).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "kernels.cuh"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while);
}

// ---------------------------------------------------------------------------
// Launch helpers. All take device pointers.
struct Ctx {
    int N;
    float *dA, *dB, *dC;
    half *hA, *hB;                          // FP16 copies for the Tensor Core paths
    cublasHandle_t blas;
};

static void launch(const std::string &k, Ctx &c)
{
    int N = c.N;
    if (k == "naive") {
        dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
        k_naive<<<grd, blk>>>(c.dA, c.dB, c.dC, N);
    } else if (k == "tiled16") {
        dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
        k_tiled<16><<<grd, blk>>>(c.dA, c.dB, c.dC, N);
    } else if (k == "tiled32") {
        dim3 blk(32, 32), grd((N + 31) / 32, (N + 31) / 32);
        k_tiled<32><<<grd, blk>>>(c.dA, c.dB, c.dC, N);
    } else if (k == "regblock") {
        dim3 grd(N / 128, N / 128);
        k_regblock<128, 128, 8, 8, 8><<<grd, 256>>>(c.dA, c.dB, c.dC, N);
    } else if (k == "vec4") {
        dim3 grd(N / 128, N / 128);
        k_vec4<128, 128, 8, 8, 8><<<grd, 256>>>(c.dA, c.dB, c.dC, N);
    } else if (k == "wmma") {
        dim3 blk(128, 4);                   // 4 x 4 warps, one 16x16 tile each
        dim3 grd((N / 16 + 3) / 4, (N / 16 + 3) / 4);
        k_wmma<<<grd, blk>>>(c.hA, c.hB, c.dC, N);
    } else if (k == "cublas") {
        // cuBLAS is column-major. Row-major C = A*B is column-major C^T = B^T A^T,
        // which is what passing B before A computes.
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                 &alpha, c.dB, N, c.dA, N, &beta, c.dC, N));
    } else if (k == "cublas_fp16") {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                  &alpha, c.hB, CUDA_R_16F, N, c.hA, CUDA_R_16F, N,
                                  &beta, c.dC, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        fprintf(stderr, "unknown kernel %s\n", k.c_str());
        exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool needs_128(const std::string &k)
{
    return k == "regblock" || k == "vec4" || k == "wmma" || k == "cublas_fp16";
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <naive|tiled16|tiled32|regblock|vec4|wmma|cublas|cublas_fp16|all> N [reps]\n", argv[0]);
        return 1;
    }
    std::string which = argv[1];
    int N = atoi(argv[2]);
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    std::vector<std::string> kernels = {"naive", "tiled16", "tiled32", "regblock", "vec4",
                                        "wmma", "cublas", "cublas_fp16"};
    if (which != "all") kernels = {which};

    size_t n = (size_t)N * N, bytes = n * sizeof(float);
    std::vector<float> A(n), B(n), C(n), Cref(n);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        A[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        B[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }

    CUDA_CHECK(cudaFree(0));                // create the context outside any timing
    Ctx c{N, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&c.dA, bytes));
    CUDA_CHECK(cudaMalloc(&c.dB, bytes));
    CUDA_CHECK(cudaMalloc(&c.dC, bytes));
    CUDA_CHECK(cudaMalloc(&c.hA, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&c.hB, n * sizeof(half)));
    CUBLAS_CHECK(cublasCreate(&c.blas));
    CUDA_CHECK(cudaMemcpy(c.dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.dB, B.data(), bytes, cudaMemcpyHostToDevice));
    k_to_half<<<(unsigned)((n + 255) / 256), 256>>>(c.dA, c.hA, n);
    k_to_half<<<(unsigned)((n + 255) / 256), 256>>>(c.dB, c.hB, n);

    // Reference result: cuBLAS FP32.
    launch("cublas", c);
    CUDA_CHECK(cudaMemcpy(Cref.data(), c.dC, bytes, cudaMemcpyDeviceToHost));
    float refmax = 0.0f;
    for (size_t i = 0; i < n; i++) refmax = fmaxf(refmax, fabsf(Cref[i]));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    for (const auto &k : kernels) {
        if (needs_128(k) && N % 128 != 0) {
            fprintf(stderr, "skip %s: N must be a multiple of 128\n", k.c_str());
            continue;
        }
        CUDA_CHECK(cudaMemset(c.dC, 0, bytes));
        launch(k, c);                       // warm-up (also loads the kernel image)
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < reps; r++) launch(k, c);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
        double kms = total_ms / reps;

        CUDA_CHECK(cudaMemcpy(C.data(), c.dC, bytes, cudaMemcpyDeviceToHost));
        float maxerr = 0.0f;
        for (size_t i = 0; i < n; i++) maxerr = fmaxf(maxerr, fabsf(C[i] - Cref[i]));

        // End to end, starting from host memory: allocate, copy in, run, copy out, free.
        CUDA_CHECK(cudaEventRecord(e0));
        float *tA, *tB, *tC;
        CUDA_CHECK(cudaMalloc(&tA, bytes));
        CUDA_CHECK(cudaMalloc(&tB, bytes));
        CUDA_CHECK(cudaMalloc(&tC, bytes));
        CUDA_CHECK(cudaMemcpy(tA, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tB, B.data(), bytes, cudaMemcpyHostToDevice));
        Ctx t = c;
        t.dA = tA; t.dB = tB; t.dC = tC;     // FP16 paths still use the converted copies
        launch(k, t);
        CUDA_CHECK(cudaMemcpy(C.data(), tC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(tA)); CUDA_CHECK(cudaFree(tB)); CUDA_CHECK(cudaFree(tC));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float e2e_ms;
        CUDA_CHECK(cudaEventElapsedTime(&e2e_ms, e0, e1));

        double gflops = 2.0 * N * (double)N * N / (kms * 1e-3) / 1e9;
        printf("%s,%d,%.4f,%.1f,%.3f,%.2e\n", k.c_str(), N, kms, gflops, e2e_ms, maxerr / refmax);
        fflush(stdout);
    }

    cublasDestroy(c.blas);
    cudaFree(c.dA); cudaFree(c.dB); cudaFree(c.dC); cudaFree(c.hA); cudaFree(c.hB);
    return 0;
}
