#!/bin/bash
# Build our own combined RT-DETR detection h5 cache (replaces Darya's
# {train,test}_detections*.h5). ONE combined cache over the full 6276-study pool
# (rl.jsonl + eval.jsonl, deduped by study_uuid) -- dicom_uuids never collide between
# train/eval splits (documented project invariant), so a single build/detections_all.h5
# covers both.
#
# Wall-time: this build does per-frame inference (up to 16x more calls per dicom than the
# per-clip MViT cache, which itself takes ~3-4h at 6276 studies) -- expect substantially
# longer, plausibly most of a day.
# Idempotent: re-running skips dicom_uuids already present in the output h5's keys.
#SBATCH --job-name=build_detr_cache
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=myqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=128G
#SBATCH --time=36:00:00
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
# Dir with echo_prime_encoder.pt and rtdetr_cardiac_7cls.pt (the newly-downloaded RT-DETR
# weights this build needs).
export ECHOPRIME_WEIGHTS_DIR=/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights

# MIOpen writes DB files to /tmp by default; /tmp is read-only on VAST (and never to be used
# per project rules regardless). Route through a job-ID-namespaced scratch dir under .tmp_work/.
MIOPEN_DIR="$REPO/.tmp_work/miopen/detr_cache_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DIR"

STUDY_LIST="$REPO/build/cache_study_list.txt"
mkdir -p "$REPO/build" logs

# Build (or reuse) the study-list from rl.jsonl + eval.jsonl (6276 unique studies, the full
# pool). Same inline-Python approach build_echoprime_cache.sbatch already uses -- rebuilding
# is cheap and idempotent-safe since it's a pure deterministic function of the two jsonl files.
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

echo "[cache] Starting combined DETR detection cache build -> build/detections_all.h5"

"$PY" -m echoprime_track.build_detr_cache \
    --studies-dir "$ECHO_PREPROCESSED_DIR" \
    --study-list  "$STUDY_LIST" \
    --out         "$REPO/build/detections_all.h5"

echo "[cache] Done."
