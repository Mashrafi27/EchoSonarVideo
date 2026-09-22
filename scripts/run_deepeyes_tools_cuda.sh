#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$PWD/packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "${DEEPEYES_PYTHON:-$PWD/.venv_deepeyes_cuda/bin/python}" -m eval.run_deepeyes_tools \
  --backend transformers --protocol hrbench "$@"
