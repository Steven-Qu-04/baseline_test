#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mol_gtn}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu121}"

conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-2.5.1+${CUDA_TAG}.html"
python -m pip install --no-cache-dir torch-geometric rdkit lmdb pandas numpy tqdm

python -m mol_gtn.check_env --output-dir /hy-tmp/result
