#!/usr/bin/env python3
"""Handout Part 7.3, extended: matrix multiplication called from Python.

Compares, per size:
  numpy        A @ B on the CPU (multi-threaded BLAS: a strong CPU baseline)
  handout      gpu_matrix_multiply: allocate + copy + tiled kernel + copy back + free per call
  ctx_<kernel> persistent device buffers; reports the whole call and the kernel alone

    python3 python/bench_matmul.py [N ...]      -> results/python_matmul.csv
"""
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab6 import Lab6  # noqa: E402

sizes = [int(a) for a in sys.argv[1:]] or [256, 512, 1024, 2048, 4096]
gpu = Lab6()
rng = np.random.default_rng(42)
rows = []


def rel_err(C, ref):
    return float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3


print(f"{'N':>5} {'impl':>14} {'call ms':>10} {'kernel ms':>10} {'GFLOP/s':>9} {'rel err':>9}")
for N in sizes:
    A = rng.uniform(-1, 1, (N, N)).astype(np.float32)
    B = rng.uniform(-1, 1, (N, N)).astype(np.float32)
    flops = 2.0 * N ** 3
    ref, t_np = timed(lambda: A @ B)
    rows.append(("numpy", N, t_np, t_np, 0.0))
    gpu.matmul(A, B, "tiled16")                         # warm-up: context, kernel load, buffers
    C, t_h = timed(lambda: gpu.matmul_handout(A, B))
    rows.append(("handout", N, t_h, float("nan"), rel_err(C, ref)))
    for k in ("tiled16", "regblock", "vec4", "cublas"):
        if k in ("regblock", "vec4") and N % 128:
            continue
        (C, kms), t = timed(lambda: gpu.matmul(A, B, k))
        rows.append((f"ctx_{k}", N, t, kms, rel_err(C, ref)))
    for impl, n, call_ms, kms, err in rows[-6:]:
        if n != N:
            continue
        base = kms if kms == kms else call_ms            # nan check: handout has no kernel-only time
        print(f"{n:>5} {impl:>14} {call_ms:>10.3f} {kms:>10.3f} {flops / (base * 1e-3) / 1e9:>9.1f} {err:>9.1e}")

os.makedirs("results", exist_ok=True)
with open("results/python_matmul.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["impl", "N", "call_ms", "kernel_ms", "rel_err"])
    w.writerows(rows)
print("-> results/python_matmul.csv")
gpu.close()
