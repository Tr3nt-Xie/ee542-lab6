#!/usr/bin/env bash
# One-time setup of a fresh WSL2 Ubuntu 24.04 for this lab: build tools, the
# CUDA toolkit (NVIDIA's WSL repository; never a Linux driver inside WSL) and
# the Python environment.
#   bash scripts/setup_wsl.sh [cuda-version]      e.g. 13-3 (default), 12-9
# Pick a version no newer than the "CUDA Version" nvidia-smi shows on Windows.
# An RTX 50-series GPU needs 12-8 or newer.
set -euo pipefail
CUDA=${1:-13-3}
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y build-essential python3-venv wget
if [ ! -x /usr/local/cuda/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y "cuda-toolkit-$CUDA"
fi
# First line of .bashrc, so non-interactive shells (ssh host cmd) see it too.
grep -q '/usr/local/cuda/bin' ~/.bashrc || sed -i '1i export PATH=/usr/local/cuda/bin:$PATH' ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy pillow matplotlib scikit-image jupyterlab

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | tail -2
echo "Setup done. Next: make all && bin/sgemm all 2048 10"
