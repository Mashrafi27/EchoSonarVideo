#!/bin/bash
# Build the EchoPrime GRPO train and val parquets from the video cache + rl/eval jsonl.
# Run AFTER build_echoprime_cache.sbatch completes.
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

CACHE_DIR="$REPO/build/echoprime_video_cache"
mkdir -p logs

echo "[parquet] Building train parquet ..."
"$PY" -m echoprime_track.generate_grpo_parquet \
    --rl-jsonl build/rl.jsonl \
    --cache-dir "$CACHE_DIR" \
    --out build/echoprime_grpo_train.parquet \
    --one-per-study \
    --shuffle-seed 42

echo "[parquet] Building val parquet ..."
"$PY" -m echoprime_track.generate_grpo_parquet \
    --rl-jsonl build/eval.jsonl \
    --cache-dir "$CACHE_DIR" \
    --out build/echoprime_grpo_val.parquet \
    --one-per-study \
    --shuffle-seed 42

echo "[parquet] Done."
echo "  train: $(python3 -c "import pyarrow.parquet as pq; print(pq.read_table('build/echoprime_grpo_train.parquet').num_rows)") rows"
echo "  val:   $(python3 -c "import pyarrow.parquet as pq; print(pq.read_table('build/echoprime_grpo_val.parquet').num_rows)") rows"
