// Stand-alone CUDA convolution benchmark (handout Step 8.2).
//   ./conv_gpu M N [reps]      random image, box filter
// Prints impl,M,N,kernel_ms,mpix_per_s,e2e_ms,max_abs_err for the naive and
// tiled kernels; the error is against the CPU implementation.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include "conv_kernels.cuh"
// The CPU reference is compiled into this file as C++ (conv2d_cpu is valid in
// both languages); CONV_NO_MAIN drops the C program's own main().
#define CONV_NO_MAIN
#include "../conv/conv_cpu.c"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s M N [reps]\n", argv[0]); return 1; }
    int M = atoi(argv[1]), N = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 10;
    if (M <= 0 || N <= 0 || N % 2 == 0 || N > CONV_MAX_N) { fprintf(stderr, "M > 0, odd N <= %d\n", CONV_MAX_N); return 1; }
    size_t px = (size_t)M * M;
    std::vector<unsigned char> img(px);
    std::vector<float> f(N * N, 1.0f / (N * N)), out(px), ref(px);
    srand(42);
    for (auto &p : img) p = (unsigned char)(rand() & 255);
    conv2d_cpu(img.data(), ref.data(), M, f.data(), N);

    CUDA_CHECK(cudaFree(0));
    unsigned char *dImg; float *dOut;
    CUDA_CHECK(cudaMalloc(&dImg, px));
    CUDA_CHECK(cudaMalloc(&dOut, px * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dImg, img.data(), px, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.data(), N * N * sizeof(float)));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    for (int tiled = 0; tiled <= 1; tiled++) {
        conv_launch(tiled, dImg, dOut, M, N);           // warm-up
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < reps; r++) conv_launch(tiled, dImg, dOut, M, N);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float total; CUDA_CHECK(cudaEventElapsedTime(&total, e0, e1));
        double kms = total / reps;
        CUDA_CHECK(cudaMemcpy(out.data(), dOut, px * sizeof(float), cudaMemcpyDeviceToHost));
        float err = 0;
        for (size_t i = 0; i < px; i++) err = fmaxf(err, fabsf(out[i] - ref[i]));

        // End to end from host memory, as a program without a resident GPU buffer would run.
        CUDA_CHECK(cudaEventRecord(e0));
        unsigned char *tI; float *tO;
        CUDA_CHECK(cudaMalloc(&tI, px)); CUDA_CHECK(cudaMalloc(&tO, px * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(tI, img.data(), px, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.data(), N * N * sizeof(float)));
        conv_launch(tiled, tI, tO, M, N);
        CUDA_CHECK(cudaMemcpy(out.data(), tO, px * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(tI)); CUDA_CHECK(cudaFree(tO));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float e2e; CUDA_CHECK(cudaEventElapsedTime(&e2e, e0, e1));
        printf("%s,%d,%d,%.4f,%.1f,%.3f,%.2e\n", tiled ? "gpu_tiled" : "gpu_naive", M, N, kms,
               (double)px / (kms * 1e-3) / 1e6, e2e, err);
    }
    cudaFree(dImg); cudaFree(dOut);
    return 0;
}
