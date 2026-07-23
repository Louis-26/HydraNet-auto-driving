#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-hydranet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
pip install -r "${ROOT_DIR}/requirements.txt"
cd scripts
bash pretrained_weight_download.sh
echo "Environment '${ENV_NAME}' is ready."
