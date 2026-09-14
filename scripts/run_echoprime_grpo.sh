#!/usr/bin/env bash
# GRPO launch for the EchoSonar-R-matched track: frozen EchoPrime + Qwen3-8B-text, no tools,
# real vLLM serving (see echo_verl/configs/echoprime_grpo.yaml's header, docs/OPEN_ISSUES.md, and
# .claude/plans/async-sprouting-graham.md for the why -- an earlier in-process HFRollout attempt
# hit a real wall in verl's trainer, real vLLM serving of this custom architecture is what
# actually works end to end). Mirrors scripts/run_grpo.sh's environment setup -- same conda env,
# same box.
#
#   bash scripts/run_echoprime_grpo.sh
set -xeuo pipefail

REPO=${REPO:-/home/mashrafimonon/EchoSonarVideo}
cd "$REPO"

# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE avoid an is_base_mistral API check transformers 4.57
# makes even for a fully-local model.path -- without it we got 429-rate-limited before
# (scripts/run_grpo.sh's own comment); also sets HF_HOME to where Qwen3-8B is already cached.
source /hdd2/ahmedaly/echogrpo/env.sh

VENV=${VENV:-/data/ahmedaly/mashrafi_echogrpo/conda_env}
PY="$VENV/bin/python"

NGPUS=${NGPUS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-$REPO/build/echoprime_cold_start_checkpoint}
TRAIN_FILES=${TRAIN_FILES:-$REPO/build/echoprime_grpo_train.parquet}
VAL_FILES=${VAL_FILES:-$REPO/build/echoprime_grpo_val_full1215.parquet}
EXP_NAME=${EXP_NAME:-grpo-echoprime-nosft-$(date +%m%d-%H%M)}
CKPT_HOME=${CKPT_HOME:-/data/ahmedaly/mashrafi_echogrpo/checkpoints/$EXP_NAME}
mkdir -p "$CKPT_HOME" logs

export WANDB_DIR=${WANDB_DIR:-/hdd2/ahmedaly/echogrpo/wandb}
export WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-echo-grpo}
mkdir -p "$WANDB_DIR"

CONFIG_PATH="$REPO/echo_verl/configs"

"$PY" -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=echoprime_grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-8} \
    data.max_prompt_length=${MAX_PROMPT_LEN:-1024} \
    data.max_response_length=${MAX_RESP_LEN:-1024} \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.46} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-2560} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-2} \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_HOME" \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_epochs=${EPOCHS:-3} \
    trainer.save_freq=${SAVE_FREQ:-15} \
    trainer.test_freq=${TEST_FREQ:-15} \
    "$@"
