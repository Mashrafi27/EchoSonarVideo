#!/usr/bin/env bash
#SBATCH --job-name=echo_merge
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Consolidate an 8-way-sharded FSDP SFT checkpoint into a single HuggingFace model
# that vLLM can load for GRPO rollouts.
#
#   STEP=100 sbatch scripts/merge_ckpt.sbatch
#
# The merge itself is pure CPU (load shards -> reassemble -> save), but this QOS
# rejects GPU-less jobs (QOSMinGRES), so we request the minimum 1 GPU and leave it idle.
# It does NOT fit on the login node -- 8 x ~6GB shards plus a full bf16 model
# materialised for saving got OOM-killed there (exit 137) against a ~125GB box
# already 90GB in use. 256G here is deliberate headroom.
set -xeuo pipefail

REPO=${REPO:-/vast/users/mohammad.yaqub/project/EchoSonarVideo}
cd "$REPO"

PY=${PY:-$REPO/.venv-train/bin/python}
STEP=${STEP:-100}
SRC=${SRC:-$REPO/checkpoints/echo-sft/coldstart-qwen3vl8b/global_step_${STEP}}
DST=${DST:-$REPO/checkpoints/echo-sft/merged/step${STEP}}

# Same ROCm trap as the training scripts: SLURM sets ROCR_VISIBLE_DEVICES, and ray
# (imported transitively by `import verl`) hard-errors on it. Needed even though this
# job never touches the GPU.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$(dirname "$DST")" logs

"$PY" external/verl/scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$SRC" \
    --target_dir "$DST"

# The merger writes weights + config but not the tokenizer/processor files; copy the
# ones the SFT job saved alongside the shards so the merged dir is self-contained.
cp -n "$SRC"/huggingface/* "$DST"/ 2>/dev/null || true

ls -la "$DST"
du -sh "$DST"
