#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for PyraMeth.
#
# PyraMeth is a PyTorch-based Nanopore 5mC methylation CLI. Cloud Agent VMs have
# no GPU, so we install the CPU PyTorch build. torchvision must come from the same
# CPU channel as torch, otherwise timm (a dependency) imports a mismatched
# torchvision and fails with "operator torchvision::nms does not exist".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 1. System build dependencies. mappy and pysam are built from source on Python
#    3.12, which needs the Python headers, a C toolchain and the htslib/zlib libs.
if ! dpkg -s python3-dev >/dev/null 2>&1 \
   || ! dpkg -s zlib1g-dev >/dev/null 2>&1 \
   || ! dpkg -s build-essential >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        python3-venv python3-dev build-essential \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev
fi

# 2. Project virtual environment.
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. CPU PyTorch stack (torch + matching torchvision from the same channel).
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. PyraMeth (editable) and the remaining Python dependencies from requirements.txt.
python -m pip install -e .

echo "[install] PyraMeth environment ready. Activate with: source .venv/bin/activate"
