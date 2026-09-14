#!/bin/bash
#SBATCH --job-name=echo_grpo_check
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=384G
#SBATCH --time=00:30:00
#SBATCH --output=.tmp_work/rocm_validation_20260914/grpo_%j.out
#SBATCH --error=.tmp_work/rocm_validation_20260914/grpo_%j.err
set -euo pipefail
cd /vast/users/mohammad.yaqub/project/EchoSonarVideo
REPO="$PWD"
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
export VLLM_PLUGINS=echo_rocm_runtime
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_TMPDIR="$ROCM_SCRATCH_ALIAS/ray"
export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DIR="$RUN_DIR" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_ENTITY=anaatef9-mbzuai
export VERL_FILE_LOGGER_PATH="$RUN_DIR/grpo_${SLURM_JOB_ID}/metrics.jsonl"
mkdir -p "$(dirname "$VERL_FILE_LOGGER_PATH")"
export ECHO_ROCM_ALLOWED_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
export ECHO_ROCM_SCRATCH_ROOT="$ROCM_SCRATCH_MOUNT"
export PYTHONPATH="$REPO/scripts/rocm_python_startup:$REPO/packages:$REPO"
export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
export ECHO_N_PREVIEW_FRAMES=2 ECHO_PREVIEW_MAX_SIDE=160 ECHO_MAX_TOTAL_FRAMES=24
export ECHO_N_HIGHRES_FRAMES=8 ECHO_HIGHRES_MAX_SIDE=320
export ECHO_SELECT_VIEW_VIDEO=0  # validated ROCm path uses image observations (job 186304)
mkdir -p "$RAY_TMPDIR" "$ROCM_SCRATCH_ALIAS/plasma"
CACHE="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct"
REVISION=$(cat "$CACHE/refs/main")
MODEL="$CACHE/snapshots/$REVISION"
PY="$RUN_DIR/env/bin/python"
DATA="$RUN_DIR/grpo_smoke.parquet"
"$PY" -u scripts/check_rocm_compat.py
"$PY" -u scripts/check_echo_tool_images.py --model "$MODEL"
if [[ "${SKIP_FSDP_REGRESSION:-0}" != 1 ]]; then
    "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$SMOKE_GPUS" scripts/check_qwen_fsdp.py
else
    echo 'QWEN_FSDP_REGRESSION_REUSED job=186271 two_steps_passed'
fi
"$PY" -u scripts/prepare_grpo_smoke.py --output "$DATA"

ARGS=(
 algorithm.adv_estimator=grpo
 algorithm.use_kl_in_reward=False
 "data.train_files=$DATA" "data.val_files=$DATA"
 data.train_batch_size=4 data.val_batch_size=4
 data.max_prompt_length=2048 data.max_response_length=1024
 data.image_patch_size=16 data.dataloader_num_workers=0
 data.return_raw_chat=True data.return_multi_modal_inputs=False
 data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=1
 data.truncation=error data.shuffle=False
 "actor_rollout_ref.model.path=$MODEL"
 actor_rollout_ref.model.external_lib=verl_bridge.fsdp_compat
 actor_rollout_ref.model.use_remove_padding=False
 actor_rollout_ref.model.enable_gradient_checkpointing=True
 +actor_rollout_ref.model.override_config.attn_implementation=sdpa
 actor_rollout_ref.actor.strategy=fsdp2
 actor_rollout_ref.actor.optim.lr=1e-6
 actor_rollout_ref.actor.ppo_mini_batch_size=4
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
 actor_rollout_ref.actor.use_dynamic_bsz=False
 actor_rollout_ref.actor.use_torch_compile=False
 actor_rollout_ref.actor.use_kl_loss=True
 actor_rollout_ref.actor.kl_loss_coef=0.01
 actor_rollout_ref.actor.entropy_coeff=0
 actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
 actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
 actor_rollout_ref.actor.fsdp_config.offload_policy=True
 actor_rollout_ref.actor.fsdp_config.param_offload=True
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
 actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
 actor_rollout_ref.ref.fsdp_config.param_offload=True
 actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
 actor_rollout_ref.ref.use_torch_compile=False
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async
 actor_rollout_ref.rollout.tensor_model_parallel_size=1
 actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=1.0
 actor_rollout_ref.rollout.gpu_memory_utilization=0.45
 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
 actor_rollout_ref.rollout.enforce_eager=True
 actor_rollout_ref.rollout.enable_prefix_caching=False
 actor_rollout_ref.rollout.max_model_len=4096
 actor_rollout_ref.rollout.max_num_batched_tokens=4096
 actor_rollout_ref.rollout.max_num_seqs=4
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
 actor_rollout_ref.rollout.agent.num_workers=2
 "actor_rollout_ref.rollout.agent.agent_loop_config_path=$REPO/packages/verl_bridge/configs/echo_agent_loop.yaml"
 actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
 actor_rollout_ref.rollout.multi_turn.enable=True
 "actor_rollout_ref.rollout.multi_turn.tool_config_path=$REPO/packages/verl_bridge/configs/echo_tool_config.yaml"
 actor_rollout_ref.rollout.multi_turn.format=hermes
 actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3
 actor_rollout_ref.rollout.multi_turn.max_user_turns=3
 +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=TRITON_ATTN
 +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA
 '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:35,video:0}'
 "custom_reward_function.path=$REPO/packages/verl_bridge/reward.py"
 custom_reward_function.name=compute_score
 "trainer.n_gpus_per_node=$SMOKE_GPUS" trainer.nnodes=1 trainer.device=cuda
 trainer.total_epochs=2 trainer.total_training_steps=2
 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1
 'trainer.logger=[console,file,wandb]'
 trainer.project_name=echo-runtime-validation
 "trainer.experiment_name=qwen3vl8b-grpo-smoke-$SLURM_JOB_ID"
 "trainer.default_local_dir=$RUN_DIR/grpo_$SLURM_JOB_ID"
 "trainer.rollout_data_dir=$RUN_DIR/grpo_$SLURM_JOB_ID/rollouts"
 trainer.resume_mode=disable
 "ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK"
 +ray_kwargs.ray_init.object_store_memory=536870912
 "+ray_kwargs.ray_init._plasma_directory=$ROCM_SCRATCH_ALIAS/plasma"
 "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR"
)
"$PY" -m verl_bridge.main_ppo "${ARGS[@]}" --cfg job > "$RUN_DIR/grpo_${SLURM_JOB_ID}_resolved.yaml"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then echo GRPO_CONFIG_PREFLIGHT_PASS; exit 0; fi
"$PY" -u - <<'PY'
import os, torch
print('GRPO_GPU_COUNT', torch.cuda.device_count(), flush=True)
count = int(os.environ["SMOKE_GPUS"])
assert torch.cuda.device_count()==count
for i in range(count):
    assert torch.ones(16,device=f'cuda:{i}').sum().item()==16
from vllm.platforms import current_platform
uuids = [current_platform.get_device_uuid(i) for i in range(count)]
assert len(set(uuids)) == count, uuids
print('GRPO_GPU_UUIDS', uuids, flush=True)
print('GRPO_DEVICE_GATE_PASS',flush=True)
PY
"$PY" -u -m verl_bridge.main_ppo "${ARGS[@]}"

"$PY" scripts/check_grpo_smoke_result.py --run-root "$RUN_DIR" --job-id "$SLURM_JOB_ID"
