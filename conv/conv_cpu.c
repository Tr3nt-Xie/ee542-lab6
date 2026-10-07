/* CPU 2-D convolution of an M x M 8-bit grayscale image with an N x N float
 * filter (N odd). Output is float so edge filters keep their sign; borders
 * clamp to the nearest edge pixel. Correlation form, as image libraries use:
 *   out[y][x] = sum_{i,j} img[y + i - r][x + j - r] * f[i][j],  r = N / 2
 *
 *   ./conv_cpu M N [reps]      random image, box filter; prints
 *   impl,M,N,ms,mpix_per_s,checksum
 * Built twice: conv_cpu (one core) and conv_cpu_omp (all cores, -fopenmp).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void conv2d_cpu(const unsigned char *img, float *out, int M, const float *f, int N)
{
    int r = N / 2;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < M; y++)
        for (int x = 0; x < M; x++) {
            float acc = 0.0f;
            for (int i = 0; i < N; i++) {
                const unsigned char *row = &img[clampi(y + i - r, 0, M - 1) * M];
                for (int j = 0; j < N; j++)
                    acc += row[clampi(x + j - r, 0, M - 1)] * f[i * N + j];
            }
            out[y * M + x] = acc;
        }
}

#ifndef CONV_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s M N [reps]\n", argv[0]); return 1; }
    int M = atoi(argv[1]), N = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 1;
    if (M <= 0 || N <= 0 || N % 2 == 0 || reps <= 0) { fprintf(stderr, "M > 0, odd N\n"); return 1; }
    unsigned char *img = malloc((size_t)M * M);
    float *out = malloc((size_t)M * M * sizeof(float));
    float *f = malloc((size_t)N * N * sizeof(float));
    if (!img || !out || !f) { perror("malloc"); return 1; }
    srand(42);
    for (size_t i = 0; i < (size_t)M * M; i++) img[i] = (unsigned char)(rand() & 255);
    for (int i = 0; i < N * N; i++) f[i] = 1.0f / (N * N);

    conv2d_cpu(img, out, M, f, N);           /* warm-up */
    double t0 = now_s();
    for (int r = 0; r < reps; r++) conv2d_cpu(img, out, M, f, N);
    double ms = (now_s() - t0) * 1e3 / reps;
#ifdef _OPENMP
    const char *impl = "cpu_omp";
#else
    const char *impl = "cpu";
#endif
    double checksum = 0;                     /* keeps the compiler from dropping the work */
    for (size_t i = 0; i < (size_t)M * M; i += 97) checksum += out[i];
    printf("%s,%d,%d,%.3f,%.1f,%.3f\n", impl, M, N, ms, (double)M * M / (ms * 1e-3) / 1e6, checksum);
    free(img); free(out); free(f);
    return 0;
}
#endif
