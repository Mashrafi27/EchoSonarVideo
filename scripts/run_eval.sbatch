#!/usr/bin/env bash
#SBATCH --job-name=echo_eval
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:mi210:8
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Evaluate a checkpoint on the held-out echo test set, 8 shards on one node.
#
#   MODEL=checkpoints/echo-sft/merged/step100 PER_TYPE=100 sbatch scripts/run_eval.sbatch
#
# The model runs IN-PROCESS (echo_verl/eval/local_client.py), not against a served
# engine: the installed vLLM is a CUDA wheel that cannot run on MI210, and every
# container route to a ROCm build is closed on this filesystem. Episodes are fully
# independent, so sharding one process per GPU recovers most of what a served
# engine's continuous batching would have given us.
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"
PY=${PY:-$REPO/.venv-train/bin/python}

MODEL=${MODEL:-$REPO/checkpoints/echo-sft/merged/step100}
TAG=${TAG:-$(basename "$MODEL")}
NSHARDS=${NSHARDS:-8}
PER_TYPE=${PER_TYPE:-100}       # cap per question type: the corpus is 88% two types
EVAL_JSONL=${EVAL_JSONL:-build/eval.jsonl}
OUT_DIR=${OUT_DIR:-$REPO/build/eval_${TAG}_${SLURM_JOB_ID}}
WANDB_PROJECT=${WANDB_PROJECT:-echo-eval}

# ROCm trap 1: SLURM exports ROCR_VISIBLE_DEVICES; torch 2.7 ROCm hard-errors on it.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ECHO_PREPROCESSED_DIR=${ECHO_PREPROCESSED_DIR:-/vast/users/mohammad.yaqub/project/preprocessed_data}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_DIR=${WANDB_DIR:-$REPO/.wandb}
mkdir -p "$OUT_DIR" logs "$WANDB_DIR"

"$PY" scripts/check_prompt_parity.py --model "$MODEL"

pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
    # Each shard needs its OWN MIOpen kernel DB. ROCm trap 4 is a read-only SQLite
    # database; eight processes sharing one writable copy re-trips it under contention.
    (
        export HIP_VISIBLE_DEVICES=$i
        export MIOPEN_USER_DB_PATH="$REPO/.tmp_work/miopen/${SLURM_JOB_ID}_$i"
        export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
        mkdir -p "$MIOPEN_USER_DB_PATH"
        "$PY" -m echo_verl.eval.run_eval \
            --local-model "$MODEL" \
            --eval-jsonl "$EVAL_JSONL" \
            --out "$OUT_DIR/shard_$i.jsonl" \
            --per-type "$PER_TYPE" \
            --shard "$i" --num-shards "$NSHARDS" \
            --max-images "${MAX_IMAGES:-64}" \
            --temperature "${TEMPERATURE:-0.0}"
    ) > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard$i.log" 2>&1 &
    pids+=($!)
done

# Wait on each PID individually: `wait` alone returns 0 even if a shard died, which
# would silently score a partial eval as if it were complete.
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "SHARD $i FAILED -- see logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_shard$i.log"
        failed=1
    fi
done
[[ $failed -eq 0 ]] || { echo "aborting: at least one shard failed, refusing to score a partial run"; exit 1; }

cat "$OUT_DIR"/shard_*.jsonl > "$OUT_DIR/episodes.jsonl"
wc -l "$OUT_DIR/episodes.jsonl"

"$PY" -m echo_verl.eval.score_eval "$OUT_DIR/episodes.jsonl" \
    --json-out "$OUT_DIR/report.json" \
    --wandb --wandb-project "$WANDB_PROJECT" \
    --wandb-name "${TAG}-${SLURM_JOB_ID}" \
    --checkpoint "$MODEL"
