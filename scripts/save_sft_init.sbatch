#!/bin/bash
# Create the EchoPrimeQwen3 checkpoint initialised from Darya's SFT Qwen3-8B-text weights.
# Output: build/echoprime_sft_init/
# Can run in parallel with build_echoprime_cache.sbatch.
#SBATCH --job-name=save_echoprime_sft_init
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x

REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"

PY="$REPO/.venv-train/bin/python"

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/packages:$REPO"

SFT_BASE=/vast/users/mohammad.yaqub/report_generation/checkpoints/sft_think_full_ft_scratch_with_actual_thinking/checkpoint-1503

mkdir -p build/echoprime_sft_init logs

"$PY" scripts/save_sft_init_checkpoint.py \
    --sft-tokenizer    "$SFT_BASE" \
    --sft-llm          "$SFT_BASE/llm" \
    --darya-projectors "$SFT_BASE" \
    --out-dir          "$REPO/build/echoprime_sft_init"

echo "[sft_init] Checkpoint saved at build/echoprime_sft_init/"
ls -lh build/echoprime_sft_init/
