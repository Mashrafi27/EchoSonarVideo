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

export REPO=/vast/users/mohammad.yaqub/project/EchoSonarVideo
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

# MIOpen writes DB files to /tmp by default; /tmp is read-only on VAST.
MIOPEN_DIR="$REPO/.tmp_work/miopen/cache_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DIR"

OUT_DIR="$REPO/build/echoprime_video_cache"
STUDY_LIST="$REPO/build/cache_study_list.txt"
mkdir -p "$OUT_DIR" logs

# Build a study-list from rl.jsonl + eval.jsonl (6276 studies we actually need).
# preprocessed_data contains 75k+ dirs from multiple datasets; --study-list restricts
# the build to only the studies referenced in our jsonl files.
echo "[cache] Building study list from rl.jsonl + eval.jsonl ..."
"$PY" - << 'PY'
import json, os
out = os.environ["REPO"] + "/build/cache_study_list.txt"
seen = set()
for jsonl in ["build/rl.jsonl", "build/eval.jsonl"]:
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["study_uuid"])
with open(out, "w") as f:
    f.write("\n".join(sorted(seen)) + "\n")
print(f"[cache] {len(seen)} unique studies -> {out}")
PY

echo "[cache] Starting EchoPrime video cache build -> $OUT_DIR"

"$PY" -m echoprime_track.build_video_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" \
    --study-list  "$STUDY_LIST" \
    --out-dir     "$OUT_DIR" \
    --bin-size 8

echo "[cache] Done. $(ls "$OUT_DIR" | wc -l) studies cached."
