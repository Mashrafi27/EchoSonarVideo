#!/bin/bash
#SBATCH --job-name=eval_qwen3vl_plain
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:mi210:8
#SBATCH --mem=640G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x
REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"

PY="$REPO/.venv-train/bin/python"
CACHE=/vast/users/mohammad.yaqub/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct
REVISION=$(cat "$CACHE/refs/main")
MODEL="$CACHE/snapshots/$REVISION"
NSHARDS=8
OUT_DIR="$REPO/build/eval_qwen3vl8b-plain-full_${SLURM_JOB_ID}"
WANDB_PROJECT=echo-eval

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO/packages:$REPO"
export WANDB_DIR="$REPO/.wandb"
export WANDB_ENTITY=anaatef9-mbzuai

mkdir -p "$OUT_DIR" logs "$REPO/.wandb"

"$PY" scripts/check_prompt_parity.py --model "$MODEL"

pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
    MIOPEN_DIR="$REPO/.tmp_work/miopen/${SLURM_JOB_ID}_${i}"
    mkdir -p "$MIOPEN_DIR"
    (
        export HIP_VISIBLE_DEVICES=$i
        export MIOPEN_USER_DB_PATH="$MIOPEN_DIR"
        export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DIR"
        "$PY" -m eval.run_eval \
            --local-model "$MODEL" \
            --eval-jsonl build/eval.jsonl \
            --out "$OUT_DIR/shard_${i}.jsonl" \
            --prompt-mode plain \
            --shard "$i" \
            --num-shards "$NSHARDS" \
            --max-images 64 \
            --temperature 0.0
    ) &> "logs/${SLURM_JOB_ID}_shard${i}.log" &
    pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed=1
done
[[ $failed -eq 0 ]] || { echo "[eval] one or more shards failed"; exit 1; }

cat "$OUT_DIR"/shard_*.jsonl > "$OUT_DIR/episodes.jsonl"
echo "[eval] merged $(wc -l < "$OUT_DIR/episodes.jsonl") episodes -> $OUT_DIR/episodes.jsonl"

"$PY" -m eval.score_eval "$OUT_DIR/episodes.jsonl" \
    --json-out "$OUT_DIR/report.json" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "qwen3vl-base-plain-full-${SLURM_JOB_ID}" \
    --checkpoint "Qwen3-VL-8B-Instruct (base)" \
    --prompt-mode plain

"$PY" -m eval.score_per_disease "$OUT_DIR/episodes.jsonl" \
    --json-out "$OUT_DIR/per_disease.json" 2>/dev/null || true

echo "[eval] done. scores at $OUT_DIR/report.json"
