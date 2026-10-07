/* CPU baselines for N x N single-precision matrix multiplication (row-major).
 *
 *   ./matmul_cpu <variant> <N> [reps]
 *   variant: naive   - the handout's i-j-k loop
 *            ikj     - same arithmetic, loop order i-k-j (unit-stride access to B and C)
 *            omp     - ikj split across all cores with OpenMP (if compiled with -fopenmp)
 *
 * Prints one CSV line: impl,N,ms,gflops,checksum
 *
 * Two deliberate differences from the handout:
 *  - Time is wall-clock (clock_gettime), not clock(). clock() measures CPU
 *    time summed over threads, so it would report a multi-threaded run as
 *    slower than it is.
 *  - Results are reported in GFLOP/s (2*N^3 flops per multiply) so CPU and
 *    GPU numbers are directly comparable across sizes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* The handout's version. The inner loop walks B down a column: consecutive
 * iterations are N floats apart, so almost every access misses the cache. */
static void mm_naive(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

/* Same flops, loop order i-k-j: the inner loop now streams along a row of B
 * and a row of C, which the hardware prefetcher and the vectorizer like. */
static void mm_ikj(const float *A, const float *B, float *C, int N)
{
    memset(C, 0, (size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            const float *b = &B[k * N];
            float *c = &C[i * N];
            for (int j = 0; j < N; j++)
                c[j] += a * b[j];
        }
}

static void mm_omp(const float *A, const float *B, float *C, int N)
{
    memset(C, 0, (size_t)N * N * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            const float *b = &B[k * N];
            float *c = &C[i * N];
            for (int j = 0; j < N; j++)
                c[j] += a * b[j];
        }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s naive|ikj|omp N [reps]\n", argv[0]);
        return 1;
    }
    const char *variant = argv[1];
    int N = atoi(argv[2]);
    int reps = argc > 3 ? atoi(argv[3]) : 1;
    void (*fn)(const float *, const float *, float *, int) =
        !strcmp(variant, "naive") ? mm_naive :
        !strcmp(variant, "ikj")   ? mm_ikj   :
        !strcmp(variant, "omp")   ? mm_omp   : NULL;
    if (!fn || N <= 0 || reps <= 0) {
        fprintf(stderr, "bad arguments\n");
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = malloc(bytes), *B = malloc(bytes), *C = malloc(bytes);
    if (!A || !B || !C) { perror("malloc"); return 1; }
    srand(42);
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        B[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }

    fn(A, B, C, N);                          /* warm-up: page in all three matrices */
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        fn(A, B, C, N);
    double ms = (now_s() - t0) * 1e3 / reps;

    double checksum = 0;                     /* keeps the compiler from dropping the work */
    for (size_t i = 0; i < (size_t)N * N; i += N + 1)
        checksum += C[i];
    double gflops = 2.0 * N * N * (double)N / (ms * 1e-3) / 1e9;
    printf("cpu_%s,%d,%.3f,%.2f,%.6f\n", variant, N, ms, gflops, checksum);

    free(A); free(B); free(C);
    return 0;
}
