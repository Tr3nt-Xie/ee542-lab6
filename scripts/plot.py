#!/usr/bin/env python3
"""Plot results/results.csv: execution time vs N (log-log) and GFLOP/s per
implementation at the largest common N, with cuBLAS as the reference.

    python3 scripts/plot.py [results/results.csv]
"""
import csv, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
rows = [r for r in csv.DictReader(open(path)) if r["kernel_ms"]]
series = collections.defaultdict(list)
for r in rows:
    series[r["impl"]].append((int(r["N"]), float(r["kernel_ms"]), float(r["gflops"])))

order = ["cpu_naive", "cpu_ikj", "cpu_omp", "naive", "tiled16", "tiled32",
         "regblock", "vec4", "cublas", "wmma", "cublas_fp16"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name in [n for n in order if n in series]:
    pts = sorted(series[name])
    ax1.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
ax1.set_xscale("log", base=2); ax1.set_yscale("log")
ax1.set_xlabel("N"); ax1.set_ylabel("time per multiply (ms, log)")
ax1.set_title("Execution time vs matrix size"); ax1.grid(True, which="both", alpha=.3)
ax1.legend(fontsize=8)

nmax = max(n for n, *_ in series.get("cublas", [(0, 0, 0)]))
ref = {n: g for n, _, g in series.get("cublas", [])}.get(nmax)
names, gf = [], []
for name in [n for n in order if n in series]:
    at = [g for n, _, g in series[name] if n == nmax]
    if at:
        names.append(name); gf.append(at[0])
bars = ax2.bar(names, gf)
for b, g in zip(bars, gf):
    label = f"{g:.0f}" + (f"\n{100 * g / ref:.0f}%" if ref else "")
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), label, ha="center", va="bottom", fontsize=8)
ax2.set_ylabel("GFLOP/s"); ax2.set_title(f"Throughput at N = {nmax} (% of cuBLAS FP32)")
ax2.tick_params(axis="x", rotation=45)
fig.tight_layout()
out = path.rsplit(".", 1)[0] + ".png"
fig.savefig(out, dpi=140)
print("wrote", out)
