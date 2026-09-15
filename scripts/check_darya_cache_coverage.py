"""Diagnostic: report how much of this project's dicom_uuid pool is present in
Darya's h5 caches (see docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md
section 1). This is informational, not a pass/fail gate -- partial coverage is
expected. Darya's own SFT (report_generation/sft_thinking/dataset.py::__getitem__)
already tolerates a missing dicom_uuid by silently omitting that view's clip-token
block, and Task 7's view-block construction replicates that same guard. Always
exits 0; run it to see the real numbers, not to decide whether to proceed.

There is no standalone `frames.jsonl` in this repo. This project's own
train/eval dicom pool is `build/rl.jsonl` (train) / `build/eval.jsonl` (eval)
-- confirmed by reading packages/echoprime_track/build_video_cache.py (which
sources dicoms from the ECHO_PREPROCESSED_DIR tree directly, not a jsonl) and
scripts/build_preprocessed_tree.py (which builds that same tree from these
two VQA jsonl files' study_uuids). Each rl.jsonl/eval.jsonl record has an
`overview.views` list of {"view", "frame", "frame_count"}, where "frame" is a
path shaped `.../<study_uuid>/di-<dicom>_<View>/<n>.png` -- the dicom_uuid is
the part of the frame's parent-directory name before the first "_", per this
project's own `di-<dicom_uuid>_<View>` convention
(packages/data_core/data/views.py:parse_clip_dirname).

Usage:
    python scripts/check_darya_cache_coverage.py \
        --source-jsonl build/rl.jsonl \
        --clip-h5 /vast/users/mohammad.yaqub/report_generation/data/clip_tokens_train.h5 \
        --detr-h5 /vast/users/mohammad.yaqub/report_generation/data/train_detections.h5

Run once for the train split and once for the eval split (swap in
clip_tokens_test.h5 / test_detections_merged.h5 -- note the real on-disk name
has a "_merged" suffix the caller must pass explicitly -- and build/eval.jsonl).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages"))
from echoprime_track.coverage import missing_dicom_uuids


def _needed_dicom_uuids(source_jsonl: str) -> set:
    needed = set()
    with open(source_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for view in rec["overview"]["views"]:
                clip_dirname = os.path.basename(os.path.dirname(view["frame"]))
                dicom_uuid = clip_dirname.split("_", 1)[0]
                needed.add(dicom_uuid)
    return needed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-jsonl", required=True,
                     help="build/rl.jsonl or build/eval.jsonl (this project's real "
                          "train/eval pool -- see module docstring)")
    ap.add_argument("--clip-h5", required=True)
    ap.add_argument("--detr-h5", required=True)
    args = ap.parse_args(argv)

    needed = _needed_dicom_uuids(args.source_jsonl)
    print(f"[coverage] {len(needed)} dicom_uuids needed from {args.source_jsonl}", flush=True)

    with h5py.File(args.clip_h5, "r") as f:
        clip_keys = set(f.keys())
    missing_clip = missing_dicom_uuids(needed, clip_keys)
    print(f"[coverage] clip h5: {len(clip_keys)} keys, "
          f"{len(missing_clip)}/{len(needed)} needed dicoms MISSING", flush=True)
    if missing_clip:
        print(f"[coverage] example missing clip dicom_uuids: {list(missing_clip)[:10]}", flush=True)

    with h5py.File(args.detr_h5, "r") as f:
        detr_keys = set(f.keys())
    missing_detr = missing_dicom_uuids(needed, detr_keys)
    print(f"[coverage] detr h5: {len(detr_keys)} keys, "
          f"{len(missing_detr)}/{len(needed)} needed dicoms MISSING "
          "(fine to have gaps here -- absent detections just means no "
          "'Structure features:' block for that view, not a hard failure)",
          flush=True)

    print(f"[coverage] clip-token coverage: {len(needed) - len(missing_clip)}/{len(needed)} "
          f"({100 * (len(needed) - len(missing_clip)) / len(needed):.1f}%) -- informational "
          "only. Partial coverage is EXPECTED: Darya's own SFT (report_generation/sft_thinking/"
          "dataset.py::__getitem__) already tolerates a missing dicom_uuid by silently omitting "
          "that view's clip-token block (keeps the view-name text line) -- Task 7's view-block "
          "construction replicates that same guard. This is not a pass/fail gate.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
