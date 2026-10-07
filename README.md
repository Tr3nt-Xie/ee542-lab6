# EE 542 Lab 6 — High-performance CUDA matrix multiplication

From a C triple loop to Tensor Cores: CPU baselines, the handout's naïve and
shared-memory CUDA kernels, cuBLAS, and extensions that close most of the gap
to cuBLAS. Every result is checked against cuBLAS and reported in GFLOP/s.

| Implementation | File | What it adds |
|---|---|---|
| `cpu_naive` | `cpu/matmul_cpu.c` | handout's i-j-k loop |
| `cpu_ikj` | `cpu/matmul_cpu.c` | loop order i-k-j: unit-stride access, vectorizable |
| `cpu_omp` | `cpu/matmul_cpu.c` | i-k-j on all cores (OpenMP) — the fair CPU baseline |
| `naive` | `gpu/sgemm.cu` | handout: one thread per output element |
| `tiled16`, `tiled32` | `gpu/sgemm.cu` | handout: shared-memory tiling |
| `regblock` | `gpu/sgemm.cu` | **extension**: 128×128 block tile, 8×8 outputs per thread in registers |
| `vec4` | `gpu/sgemm.cu` | **extension**: + 128-bit `float4` loads/stores, transposed A tile |
| `wmma` | `gpu/sgemm.cu` | **extension**: Tensor Cores, FP16 inputs / FP32 accumulate |
| `cublas`, `cublas_fp16` | `gpu/sgemm.cu` | references (FP32, and FP16 Tensor Cores) |

Parts 7–8:

| Piece | File | What |
|---|---|---|
| shared library | `python/lab6lib.cu` → `bin/liblab6.so` | handout's `gpu_matrix_multiply`, plus a context that keeps device buffers alive across calls (**extension**) and reports kernel time separately; 2-D convolution |
| Python wrapper | `python/lab6.py` | `Lab6().matmul(A, B, kernel)`, `.conv2d(img, filt)` |
| matmul from Python | `python/bench_matmul.py` | NumPy vs handout call vs persistent buffers, per size |
| convolution | `conv/conv_cpu.c`, `gpu/conv_kernels.cuh`, `gpu/conv_gpu.cu` | CPU (1 core, OpenMP), CUDA naive and shared-memory tiled with halo, filter in constant memory |
| filters and timing | `python/conv_demo.py` | blur, Gaussian, sharpen, Sobel edges, Laplacian, emboss on sample images, each checked against NumPy; C program vs CUDA executable vs Python + library for 3 image sizes × 3 filter sizes |

## Run

Two notebooks for Google Colab (free T4; set the runtime to T4 GPU):

- `notebook/lab6_learn.ipynb` — **learn by building**: every program (device query, CPU
  loops, each CUDA kernel, the shared library, convolution) is written out in its own
  cell with an explanation, a prediction to make and an experiment to try
  ([open in Colab](https://colab.research.google.com/github/Tr3nt-Xie/ee542-lab6/blob/main/notebook/lab6_learn.ipynb))
- `notebook/lab6.ipynb` — **just run it**: builds the repository and produces every result

On a GPU VM:

```bash
make all                     # ARCH=sm_75 for a T4 if -arch=native is unsupported
bin/sgemm all 2048 10        # every GPU implementation at one size
bash scripts/run_all.sh      # full sweep -> results/results.csv
python3 scripts/plot.py      # -> results/results.png
python3 python/bench_matmul.py
python3 python/conv_demo.py  # -> results/conv/*.png, results/conv_perf.csv
```

`bin/sgemm` prints `impl,N,kernel_ms,gflops,e2e_ms,max_rel_err`.
`kernel_ms` is the kernel alone (CUDA events, mean of repetitions after a
warm-up); `e2e_ms` is one call from host memory including allocation and
PCIe copies. The gap between the two is the accelerator overhead the
handout asks about.

See `LEARN.md` for what each step changes and what to look for in the numbers.
