#!/usr/bin/env bash
# Run every implementation over a range of sizes and write results/results.csv.
#   bash scripts/run_all.sh                 # default sizes
#   SIZES="512 1024 2048 4096" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"256 512 1024 2048 4096"}
CPU_MAX=${CPU_MAX:-2048}                 # the naive CPU loop takes minutes beyond this
mkdir -p results
OUT=results/results.csv
echo "impl,N,kernel_ms,gflops,e2e_ms,max_rel_err" > "$OUT"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader | tee results/gpu.txt
for N in $SIZES; do
  if [ "$N" -le "$CPU_MAX" ]; then
    for v in naive ikj; do
      bin/matmul_cpu $v "$N" 1 | awk -F, '{print $1","$2","$3","$4",,"}' | tee -a "$OUT"
    done
    OMP_NUM_THREADS=$(nproc) bin/matmul_cpu_omp omp "$N" 1 | awk -F, '{print $1","$2","$3","$4",,"}' | tee -a "$OUT"
  fi
  bin/sgemm all "$N" 10 | tee -a "$OUT"
done
echo "-> $OUT"
