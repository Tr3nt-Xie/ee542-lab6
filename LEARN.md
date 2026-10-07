# Learning guide: why each version is faster

Work through the implementations in order. For each one: read the kernel,
predict the effect, run it, then explain the number.

## 0. Units

An N×N multiply does N³ multiply-adds = **2N³ flops**. Divide by the time to
get FLOP/s; this is the only fair way to compare sizes and implementations.
Reference points for an NVIDIA T4: about **8 TFLOP/s FP32** peak, **65 TFLOP/s
FP16 on Tensor Cores**, **320 GB/s** memory bandwidth, PCIe 3.0 x16 at ~12 GB/s.

**Arithmetic intensity** = flops per byte moved from memory. A kernel is
*memory-bound* if intensity × bandwidth < peak flops, *compute-bound*
otherwise. This one idea (the roofline model) explains almost every result
below.

## 1. CPU: cache behaviour beats arithmetic

`cpu_naive` reads `B[k*N + j]` with k changing fastest: each access is N
floats away from the previous one, so nearly every load misses the cache.
`cpu_ikj` performs exactly the same flops with `j` innermost: rows of B and C
stream through cache and the compiler emits SIMD instructions. Measured on an
M-series Mac at N=1024: 2.7 → 32.8 GFLOP/s, same checksum. `cpu_omp` then
uses every core. Compare the GPU against this, not against `cpu_naive`.

## 2. Naïve CUDA: parallel, but starved

One thread per output element, N² threads. Each thread does 2N flops and
loads 2N floats from global memory: intensity 0.25 flop/byte. At 320 GB/s
that caps the kernel near 80 GFLOP/s *if* every load went to DRAM; caches
and the broadcast of A within a warp do better, but it is far below 8 TFLOP/s.
Look up **coalescing**: why threadIdx.x indexing the column of C matters.

## 3. Shared-memory tiling: reuse

A T×T tile of A and B is loaded once and each value used T times, so global
traffic drops by T (16 or 32). The two `__syncthreads()` are both required:
the first so no thread reads a tile before it is fully written, the second
so no thread overwrites it while others still read. Now the bottleneck moves
to **shared memory**: the inner loop does one FMA per two shared loads.

## 4. Register blocking (extension)

Each thread computes an 8×8 block of C kept in 64 registers. Per k-step it
reads 8 values of A and 8 of B from shared memory and does 64 FMAs: 4 FMAs
per shared load instead of 0.5. This is the step that should bring the
kernel from a fraction of cuBLAS to most of it. Trade-off: 64+ registers per
thread lowers **occupancy** (fewer resident warps), which is fine as long as
each warp has enough independent work to hide latency.

## 5. Vectorized memory (extension)

`float4` moves 16 bytes per instruction instead of 4, cutting load/store
instruction count by 4. Storing the A tile transposed makes the inner loop
read both tiles with unit stride. Expect a smaller, but measurable, gain.

## 6. Tensor Cores (extension)

`wmma` hands 16×16×16 FP16 matrix multiplies to the Tensor Cores. Even this
simple version (no shared memory) can beat FP32 kernels, and `cublas_fp16`
shows the ceiling. Watch `max_rel_err`: FP16 inputs carry ~3 decimal digits,
so the error rises from ~1e-6 to ~1e-3. Faster is not free.

## 7. Overhead (handout Part 5)

Compare `kernel_ms` with `e2e_ms`. Copying 3·4N² bytes over PCIe costs about
1 ms at N=1024 and 16 ms at N=4096, plus allocation. For small N this
dominates and the GPU can lose to the CPU; find the **crossover N** where the
end-to-end GPU time beats `cpu_omp`. Lesson for library design: keep data on
the device across calls.

## 8. Calling the GPU from Python (Part 7)

The handout's `gpu_matrix_multiply` pays `cudaMalloc` + two host-to-device
copies + one device-to-host copy + `cudaFree` on every call. The context API
keeps buffers alive, so repeated calls pay only the copies. Compare the
`call ms` and `kernel ms` columns of `bench_matmul.py`: for small N almost all
of the call is overhead, and NumPy's CPU BLAS wins. That is the honest answer
to "when is the accelerator worth it".

## 9. Convolution (Part 8)

Each output pixel needs N² multiply-adds but its inputs overlap heavily with
its neighbours'. The naive kernel re-reads every input pixel up to N² times
from global memory (the L1/L2 caches absorb much of it); the tiled kernel
loads a (16 + N − 1)² window once into shared memory, halo included. The
filter lives in **constant memory** because every thread in a warp reads the
same tap at the same time, which constant memory broadcasts. Convolution with
small filters is memory-bound, so the gain from tiling grows with N.
Separable filters (box, Gaussian) can be done as two 1-D passes, N instead of
N² work per pixel: a natural next extension.

## Questions to be able to answer

1. Why is cuBLAS still faster? (hand-tuned per architecture, double
   buffering, warp-level tiling, assembly-level scheduling, autotuned tile
   sizes per N)
2. Why does the GPU advantage grow with N? (O(N³) work vs O(N²) transfer)
3. Which implementations are memory-bound and which compute-bound, and how
   do you know? (intensity vs the roofline; Nsight Compute's "Speed of
   Light" section)
