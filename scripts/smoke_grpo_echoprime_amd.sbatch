#!/bin/bash
# 2-step EchoPrime GRPO smoke test on AMD MI210 (ROCm).
# Validates: model load, vLLM serving, EchoPrimeToolAgentLoop, LoRA FSDP actor,
# 2 full train steps + weight sync. Inspect rewards/truncation before a full run.
#
# Chain:
#   sbatch --dependency=afterok:<parquet_job> scripts/smoke_grpo_echoprime_amd.sbatch
# Process success alone does not establish a usable GRPO reward signal.
#SBATCH --job-name=echo_ep_smoke
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=384G
#SBATCH --time=01:00:00
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

SMOKE_DIR="$REPO/.tmp_work/smoke_echoprime_${SLURM_JOB_ID}"
export WANDB_DIR="$SMOKE_DIR"
export WANDB_ENTITY=anaatef9-mbzuai
export WANDB_CACHE_DIR="$SMOKE_DIR/wandb_cache"
export VERL_FILE_LOGGER_PATH="$SMOKE_DIR/metrics.jsonl"
export ECHO_ROCM_ALLOWED_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
export ECHO_ROCM_SCRATCH_ROOT="$ROCM_SCRATCH_MOUNT"
export PYTHONPATH="$REPO/scripts/rocm_python_startup:$REPO/packages:$REPO"

export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
export ECHOPRIME_VIDEO_CACHE_DIR="$REPO/build/echoprime_video_cache"
export ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights

mkdir -p "$SMOKE_DIR" logs "$RAY_TMPDIR" "$ROCM_SCRATCH_ALIAS/plasma"

PY="$RUN_DIR/env/bin/python"
MODEL="$REPO/build/echoprime_sft_init"

# Sanity checks before touching GPUs.
[[ -f "$MODEL/config.json" ]]                        || { echo "ERROR: SFT-init checkpoint missing"; exit 1; }
[[ -f "$REPO/build/echoprime_grpo_train.parquet" ]]  || { echo "ERROR: train parquet missing";       exit 1; }
[[ -d "$ECHOPRIME_VIDEO_CACHE_DIR" ]] && \
    [[ $(ls "$ECHOPRIME_VIDEO_CACHE_DIR" | wc -l) -gt 0 ]] \
                                                     || { echo "ERROR: video cache empty";            exit 1; }

echo "[smoke] GPU check ..."
"$PY" -u -c "
import torch
n = torch.cuda.device_count()
print('GPUs:', n)
assert n == int('$SMOKE_GPUS'), f'Expected $SMOKE_GPUS GPUs, got {n}'
for i in range(n):
    assert torch.ones(8, device=f'cuda:{i}').sum().item() == 8
print('[smoke] GPU gate PASS')
"

"$PY" -m pytest -q \
    packages/verl_bridge/tests/test_rocm_triton_cache.py \
    packages/echoprime_track/tests/test_echoprime_tool_agent_loop.py \
    packages/echoprime_track/tests/test_grpo_prompt_contract.py \
    packages/echoprime_track/tests/test_generate_grpo_parquet.py

"$PY" -m echoprime_track.check_grpo_prompts \
    build/echoprime_grpo_train.parquet build/echoprime_grpo_val.parquet \
    --tokenizer "$MODEL" --max-prompt-length 3584

CONFIG_PATH="$REPO/packages/verl_bridge/configs"

ARGS=(
    --config-path="$CONFIG_PATH"
    --config-name=echoprime_grpo
    "actor_rollout_ref.model.path=$MODEL"
    "data.train_files=$REPO/build/echoprime_grpo_train.parquet"
    "data.val_files=$REPO/build/echoprime_grpo_val.parquet"
    data.train_batch_size=4
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    data.max_prompt_length=3584
    data.max_response_length=2048
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=TRITON_ATTN"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.46
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.max_model_len=6144
    actor_rollout_ref.rollout.trace.backend=weave
    actor_rollout_ref.rollout.trace.token2text=True
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=1
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
    # 2 steps only -- enough to exercise model load, rollout, weight sync, actor update.
    trainer.total_training_steps=2
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    'trainer.logger=[console,file,wandb]'
    trainer.project_name=echo-grpo-smoke
    "trainer.experiment_name=echoprime-smoke-${SLURM_JOB_ID}"
    "trainer.default_local_dir=$SMOKE_DIR"
    "trainer.rollout_data_dir=$SMOKE_DIR/rollouts"
    "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK"
    +ray_kwargs.ray_init.object_store_memory=4294967296
    "+ray_kwargs.ray_init._plasma_directory=$ROCM_SCRATCH_ALIAS/plasma"
    "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR"
)

echo "[smoke] Config preflight ..."
"$PY" -u -m verl_bridge.main_ppo "${ARGS[@]}" --cfg job > "$SMOKE_DIR/resolved.yaml" 2>&1 \
    && echo "[smoke] Config preflight PASS" \
    || { echo "[smoke] Config preflight FAIL"; cat "$SMOKE_DIR/resolved.yaml"; exit 1; }

echo "[smoke] Running 2-step EchoPrime GRPO ..."
"$PY" -u -m verl_bridge.main_ppo "${ARGS[@]}"

echo "[smoke] PASS -- EchoPrime GRPO pipeline validated on ROCm."
echo "[smoke] Logs: $SMOKE_DIR"
