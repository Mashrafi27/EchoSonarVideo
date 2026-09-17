#!/bin/bash
# Build the EchoPrime GRPO train and val parquets from our combined self-inference h5 caches +
# rl/eval jsonl. Run AFTER build_clip_grid_cache.sbatch and build_detr_cache.sbatch complete
# (build/clip_tokens_all.h5, build/detections_all.h5 -- replaces Darya's precomputed caches,
# see docs/superpowers/plans/2026-09-16-echoprime-self-inference.md).
#
# Outputs:
#   build/echoprime_grpo_train.parquet  (~5061 rows, one per training study)
#   build/echoprime_grpo_val.parquet    (~1215 rows, one per test study)
#SBATCH --job-name=build_echoprime_parquet
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x

REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"

PY="$REPO/.venv-train/bin/python"

export PYTHONPATH="$REPO/packages:$REPO"
# No GPU needed; torch.load for cache files still imports torch.
export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CLIP_H5="$REPO/build/clip_tokens_all.h5"
DETR_H5="$REPO/build/detections_all.h5"
mkdir -p logs

echo "[parquet] Building train parquet ..."
"$PY" -m echoprime_track.generate_grpo_parquet \
    --rl-jsonl build/rl.jsonl \
    --clip-h5 "$CLIP_H5" \
    --detr-h5 "$DETR_H5" \
    --out build/echoprime_grpo_train.parquet \
    --one-per-study \
    --shuffle-seed 42

echo "[parquet] Building val parquet ..."
"$PY" -m echoprime_track.generate_grpo_parquet \
    --rl-jsonl build/eval.jsonl \
    --clip-h5 "$CLIP_H5" \
    --detr-h5 "$DETR_H5" \
    --out build/echoprime_grpo_val.parquet \
    --one-per-study \
    --shuffle-seed 42

echo "[parquet] Done."
echo "  train: $("$PY" -c "import pyarrow.parquet as pq; print(pq.read_table('build/echoprime_grpo_train.parquet').num_rows)") rows"
echo "  val:   $("$PY" -c "import pyarrow.parquet as pq; print(pq.read_table('build/echoprime_grpo_val.parquet').num_rows)") rows"
