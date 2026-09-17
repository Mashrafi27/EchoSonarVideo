#!/bin/bash
# EchoPrime + Qwen3-8B-text GRPO training on AMD MI210 (ROCm).
# Initialised from Darya's SFT checkpoint (build/echoprime_sft_init/).
#
# Prerequisites (run in order before this):
#   1. sbatch scripts/build_echoprime_cache.sbatch
#   2. sbatch scripts/build_echoprime_parquet.sbatch
#   3. sbatch scripts/save_sft_init.sbatch
#
# Usage:
#   sbatch scripts/run_grpo_echoprime_amd.sbatch
#SBATCH --job-name=echo_grpo_sft_init
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x

REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"
RUN_DIR="$REPO/.tmp_work/rocm_validation_20260914"

export SMOKE_GPUS="${SLURM_GPUS_ON_NODE:-2}"
source scripts/rocm_workspace_scratch.sh
rocm_scratch_setup "$RUN_DIR" 16G
trap rocm_scratch_cleanup EXIT

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; unset ROCR_VISIBLE_DEVICES; fi
unset LD_PRELOAD HSA_OVERRIDE_GFX_VERSION

export ROCM_PATH="$RUN_DIR/env/rocm-root/opt/rocm-7.0.2"
export LD_LIBRARY_PATH="$RUN_DIR/env/native-root/usr/lib/x86_64-linux-gnu:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export LD_PRELOAD="$RUN_DIR/env/native-root/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export PATH="$RUN_DIR/env/bin:$ROCM_PATH/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_ROCM_USE_AITER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=echo_rocm_runtime,register_echoprime_model
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_TMPDIR="$ROCM_SCRATCH_ALIAS/ray"

export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXP_NAME="grpo-echoprime-sft-init-${SLURM_JOB_ID}"
CKPT_DIR="$REPO/build/grpo_echoprime_sft_init_${SLURM_JOB_ID}"

export WANDB_DIR="$REPO/.wandb"
export WANDB_ENTITY=anaatef9-mbzuai
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export VERL_FILE_LOGGER_PATH="$CKPT_DIR/metrics.jsonl"
export ECHO_ROCM_ALLOWED_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
export ECHO_ROCM_SCRATCH_ROOT="$ROCM_SCRATCH_MOUNT"
export PYTHONPATH="$REPO/scripts/rocm_python_startup:$REPO/packages:$REPO"

# EchoPrime-specific env
export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
export ECHOPRIME_VIDEO_CACHE_DIR="$REPO/build/echoprime_video_cache"
export ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights

mkdir -p "$CKPT_DIR" logs "$REPO/.wandb" "$RAY_TMPDIR" "$ROCM_SCRATCH_ALIAS/plasma"

PY="$RUN_DIR/env/bin/python"
MODEL="$REPO/build/echoprime_sft_init"

# Quick sanity: model dir exists and has expected files.
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: $MODEL/config.json not found -- run save_sft_init.sbatch first"; exit 1; }
[[ -f "$REPO/build/echoprime_grpo_train.parquet" ]] || { echo "ERROR: train parquet missing -- run build_echoprime_parquet.sbatch first"; exit 1; }

"$PY" -u -c "import torch; print('GPU count:', torch.cuda.device_count())"

CONFIG_PATH="$REPO/packages/verl_bridge/configs"

ARGS=(
    --config-path="$CONFIG_PATH"
    --config-name=echoprime_grpo
    "actor_rollout_ref.model.path=$MODEL"
    "data.train_files=$REPO/build/echoprime_grpo_train.parquet"
    "data.val_files=$REPO/build/echoprime_grpo_val.parquet"
    data.train_batch_size=8
    data.max_prompt_length=3584
    data.max_response_length=1024
    # ROCm: TRITON_ATTN for decoder self-attn, TORCH_SDPA for mm encoder
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=TRITON_ATTN"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.46
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.max_model_len=4608
    "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_mm_embeds=true"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={clip:4000,detr:100}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.strategy=fsdp2
    "actor_rollout_ref.model.external_lib=verl_bridge.fsdp_compat"
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO/packages/verl_bridge/configs/echoprime_agent_loop.yaml"
    "trainer.n_gpus_per_node=$SMOKE_GPUS"
    trainer.nnodes=1
    trainer.device=cuda
    trainer.total_epochs=3
    trainer.save_freq=50
    trainer.test_freq=50
    'trainer.logger=[console,file,wandb]'
    trainer.project_name=echo-grpo
    "trainer.experiment_name=$EXP_NAME"
    "trainer.default_local_dir=$CKPT_DIR"
    "trainer.rollout_data_dir=$CKPT_DIR/rollouts"
    trainer.resume_mode=disable
    "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK"
    +ray_kwargs.ray_init.object_store_memory=536870912
    "+ray_kwargs.ray_init._plasma_directory=$ROCM_SCRATCH_ALIAS/plasma"
    "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR"
)

echo "[grpo] Config check ..."
"$PY" -u -m verl_bridge.main_ppo "${ARGS[@]}" --cfg job > "$CKPT_DIR/resolved.yaml" 2>&1
cat "$CKPT_DIR/resolved.yaml" | head -80

echo "[grpo] Starting GRPO training ..."
"$PY" -u -m verl_bridge.main_ppo "${ARGS[@]}"

echo "[grpo] Done. Checkpoints in $CKPT_DIR"
