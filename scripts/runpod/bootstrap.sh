#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONIOENCODING=utf-8
python3 -m venv "$REPO_ROOT/.runpod-venv"
PYTHON="$REPO_ROOT/.runpod-venv/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install numpy==2.2.6 Pillow==11.3.0 segmentation-models-pytorch==0.5.0 timm==1.0.25 huggingface-hub==1.8.0 safetensors==0.5.3
"$PYTHON" -m pip check
exec "$PYTHON" scripts/runpod/remote_run.py "$@"
