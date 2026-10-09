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

Two notebooks (they work on your own GPU and on Google Colab):

- `notebook/lab6_learn.ipynb` — **learn by building**: every program (device query, CPU
  loops, each CUDA kernel, the shared library, convolution) is written out in its own
  cell with an explanation, a prediction to make and an experiment to try
  ([open in Colab](https://colab.research.google.com/github/Tr3nt-Xie/ee542-lab6/blob/main/notebook/lab6_learn.ipynb))
- `notebook/lab6.ipynb` — **just run it**: builds the repository and produces every result

From a shell on any Linux machine with an NVIDIA GPU and the CUDA toolkit:

```bash
make all                     # ARCH=sm_120 (RTX 50xx) / sm_75 (T4) if -arch=native is unsupported
bin/sgemm all 2048 10        # every GPU implementation at one size
SIZES="256 512 1024 2048 4096 8192" bash scripts/run_all.sh   # full sweep -> results/results.csv
python3 scripts/plot.py      # -> results/results.png
python3 python/bench_matmul.py
python3 python/conv_demo.py  # -> results/conv/*.png, results/conv_perf.csv
```

### Run on your own GPU (e.g. RTX 5070 Ti)

An RTX 50-series card (Blackwell, compute capability 12.0) needs **CUDA 12.8 or
newer**; any 13.x toolkit works.

On **Windows, use WSL2** rather than native Windows: natively, nvcc needs MSVC,
the library becomes a `.dll` whose functions must be marked
`__declspec(dllexport)`, and the Makefile and bash scripts do not run. CUDA in
WSL2 uses the Windows driver and runs at near-native speed.

1. Update the **Windows** NVIDIA driver. Do **not** install a Linux NVIDIA driver
   inside WSL; the Windows driver serves WSL too.
2. In PowerShell: `wsl --install -d Ubuntu-24.04`, then open Ubuntu.
3. In Ubuntu, install the toolkit (NVIDIA's WSL repository) and check it:

   ```bash
   sudo apt update && sudo apt install -y build-essential git python3-venv
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   nvidia-smi && nvcc --version
   ```

   (Native Ubuntu: same, with the `ubuntu2404` repository instead of `wsl-ubuntu`
   and the regular Linux driver.)
4. Clone, set up Python, build, and open the notebook:

   ```bash
   git clone https://github.com/Tr3nt-Xie/ee542-lab6.git && cd ee542-lab6
   python3 -m venv .venv && . .venv/bin/activate
   pip install numpy pillow matplotlib scikit-image jupyterlab
   make all
   jupyter lab notebook/lab6_learn.ipynb    # open the printed localhost URL in a Windows browser
   ```

   VS Code with the WSL extension also opens the notebook directly. Keep the
   repository inside the Linux filesystem (`~/...`), not under `/mnt/c`, which is
   much slower to build from.

The handout recommends running once on a cloud GPU as well (Part 3): open
`notebook/lab6.ipynb` in Colab and run `bin/sgemm all 2048 10` on its T4. The
two GPUs side by side are a good result in themselves.

`bin/sgemm` prints `impl,N,kernel_ms,gflops,e2e_ms,max_rel_err`.
`kernel_ms` is the kernel alone (CUDA events, mean of repetitions after a
warm-up); `e2e_ms` is one call from host memory including allocation and
PCIe copies. The gap between the two is the accelerator overhead the
handout asks about.

See `LEARN.md` for what each step changes and what to look for in the numbers.
