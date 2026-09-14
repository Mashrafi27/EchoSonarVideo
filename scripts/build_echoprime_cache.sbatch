#!/bin/bash
# Build the frozen EchoPrime video-embedding cache: one .pt file per study.
# Must complete before build_echoprime_parquet.sbatch can run.
#
# Wall-time estimate: ~6276 studies, mViT-v2-S at bin_size=8, 1 GPU -> ~3-4 h.
# Idempotent: re-running skips already-cached studies.
#SBATCH --job-name=build_echoprime_cache
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
set -x

REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
cd "$REPO"

PY="$REPO/.venv-train/bin/python"

# ROCR_VISIBLE_DEVICES -> HIP_VISIBLE_DEVICES (SLURM sets ROCR, torch wants HIP).
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    unset ROCR_VISIBLE_DEVICES
fi

export HF_HOME=/vast/users/mohammad.yaqub/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/packages:$REPO"
export ECHO_PREPROCESSED_DIR=/vast/users/mohammad.yaqub/project/preprocessed_data
# Path to echo_prime_encoder.pt (frozen mViT-v2-S weights).
export ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights

OUT_DIR="$REPO/build/echoprime_video_cache"
mkdir -p "$OUT_DIR" logs

echo "[cache] Starting EchoPrime video cache build -> $OUT_DIR"

"$PY" -m echoprime_track.build_video_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" \
    --out-dir "$OUT_DIR" \
    --bin-size 8

echo "[cache] Done. $(ls "$OUT_DIR" | wc -l) studies cached."
