#!/bin/bash
# Rebuild build/rl.jsonl and build/eval.jsonl from the full VQA source files
# (128,215 train rows / 5,061 studies; 31,209 test rows / 1,215 studies).
#
# Source files are in Darya's data dir (shared VAST, read-only for us).
# Also rebuilds rl_train.parquet and rl_val.parquet for the Qwen3-VL track.
#
# Run this before build_echoprime_cache.sbatch so the study list is complete.
#SBATCH --job-name=rebuild_rl_jsonl
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x

export REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"

PY="$REPO/.venv-train/bin/python"

export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/packages:$REPO"
export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
export ECHO_VQA_TRAIN=/vast/users/mohammad.yaqub/report_generation/data/train_vqa_with_thinking.jsonl
export ECHO_VQA_TEST=/vast/users/mohammad.yaqub/report_generation/data/test_vqa.jsonl
# data_core._PROJ resolves to packages/ not repo root (the comment in config.py is wrong).
# Set ECHO_OUT_DIR explicitly so outputs land in build/ where everything else reads from.
export ECHO_OUT_DIR="$REPO/build"

mkdir -p build logs

echo "[rebuild] Building rl.jsonl from full train_vqa_with_thinking.jsonl ..."
"$PY" -m data_core.cli build-rl --split all
echo "[rebuild] rl.jsonl: $(wc -l < build/rl.jsonl) rows"

echo "[rebuild] Building eval.jsonl from full test_vqa.jsonl ..."
"$PY" -m data_core.cli build-eval --split all
echo "[rebuild] eval.jsonl: $(wc -l < build/eval.jsonl) rows"

# Also rebuild the Qwen3-VL track parquets from the new full rl.jsonl.
echo "[rebuild] Rebuilding rl_train.parquet ..."
"$PY" packages/verl_bridge/generate_trainset.py \
    --rl-jsonl build/rl.jsonl \
    --out build/rl_train.parquet

echo "[rebuild] Rebuilding rl_val.parquet ..."
"$PY" packages/verl_bridge/generate_trainset.py \
    --rl-jsonl build/eval.jsonl \
    --out build/rl_val.parquet

echo "[rebuild] Done."
echo "  rl.jsonl:       $(wc -l < build/rl.jsonl) rows"
echo "  eval.jsonl:     $(wc -l < build/eval.jsonl) rows"
