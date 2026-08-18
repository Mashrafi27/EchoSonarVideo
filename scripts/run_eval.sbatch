#!/usr/bin/env bash
#SBATCH --job-name=echo_eval
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Serve a model with vLLM, run the echo agentic loop over the held-out test set,
# and score the result.
#
#   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct TAG=base   PER_TYPE=40 sbatch scripts/run_eval.sbatch
#   MODEL_PATH=$PWD/checkpoints/echo-sft/merged/step100 TAG=step100 PER_TYPE=40 sbatch scripts/run_eval.sbatch
#
# PER_TYPE caps episodes per question type. The corpus is 88% short-answer, so an
# uncapped sample would barely touch full_report/conclusion -- the types
# EchoSonar-R reports BLEU-4 and GREEN on.
#
# Substrate: docs/TRAINING_ENV.md 2a (ROCm/MI210 overlay venv).
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"

PY=${PY:-$REPO/.venv-train/bin/python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
TAG=${TAG:-base}
PER_TYPE=${PER_TYPE:-40}
PORT=${PORT:-8${SLURM_JOB_ID: -3}}
NGPUS=${NGPUS:-2}
OUT=${OUT:-$REPO/build/eval_${TAG}.jsonl}

# ROCm traps, same as the training scripts (docs/TRAINING_ENV.md):
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-$REPO/.tmp_work/miopen/${SLURM_JOB_ID:-local}}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" logs
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ECHO_PREPROCESSED_DIR=${ECHO_PREPROCESSED_DIR:-/vast/users/mohammad.yaqub/project/preprocessed_data}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VLLM_USE_AITER=${VLLM_USE_AITER:-0}      # MI210 has no FP8/AITER path

"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name echo \
    --port "$PORT" \
    --tensor-parallel-size "$NGPUS" \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 40}' \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --disable-log-requests &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for the server, but give up rather than hang the whole allocation.
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[eval] server up after ${i}0s"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[eval] SERVER DIED during startup"; exit 1
    fi
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[eval] server never came up"; exit 1; }

"$PY" -m echo_verl.eval.run_eval \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model echo \
    --per-type "$PER_TYPE" \
    --out "$OUT"

"$PY" -m echo_verl.eval.score_eval "$OUT" --json-out "${OUT%.jsonl}_metrics.json"
