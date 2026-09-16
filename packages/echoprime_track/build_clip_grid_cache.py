"""Build our own clip-token grid h5 cache, replacing Darya's clip_tokens_{train,test}.h5.
Same per-dicom (393, 768) schema (darya_cache.load_clip_tokens reads it unchanged) -- see
docs/superpowers/specs/2026-09-16-echoprime-self-inference-design.md.

Idempotent/resumable: skips a dicom_uuid already present in the output h5's keys (matches
build_video_cache.py's per-study resumability, at dicom granularity here since that's this
cache's real key).

Usage:
    python -m echoprime_track.build_clip_grid_cache --studies-dir "$ECHO_PREPROCESSED_DIR" \\
        --out build/clip_tokens_train.h5
"""
import argparse
import glob
import os
import time

import h5py
import torch

from data_core.data.views import parse_clip_dirname
from echoprime_track.encoder import embed_clip_grid, load_frozen_encoder
from echoprime_track.preprocess import clip_to_tensor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study-list", default=None,
                     help="optional file, one study_uuid per line (default: every subdir)")
    ap.add_argument("--bin-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.study_list:
        with open(args.study_list) as fh:
            study_ids = [l.strip() for l in fh if l.strip()]
    else:
        study_ids = sorted(
            d for d in os.listdir(args.studies_dir)
            if os.path.isdir(os.path.join(args.studies_dir, d)))
    if args.limit:
        study_ids = study_ids[:args.limit]

    print(f"[clip_grid_cache] {len(study_ids)} studies queued -> {args.out}", flush=True)
    encoder = load_frozen_encoder()

    done = skipped = failed = 0
    t0 = time.monotonic()
    with h5py.File(args.out, "a") as out_h5:
        for i, study_uuid in enumerate(study_ids):
            study_dir = os.path.join(args.studies_dir, study_uuid)
            clip_dirs = sorted(
                d for d in glob.glob(os.path.join(study_dir, "di-*")) if os.path.isdir(d))
            for clip_dir in clip_dirs:
                dicom_uuid, _view = parse_clip_dirname(os.path.basename(clip_dir))
                if dicom_uuid in out_h5:
                    skipped += 1
                    continue
                try:
                    tensor = clip_to_tensor(clip_dir).unsqueeze(0)  # (1, 3, 16, 224, 224)
                    grid = embed_clip_grid(encoder, tensor, bin_size=args.bin_size)[0]  # (393,768)
                    out_h5.create_dataset(f"{dicom_uuid}/tokens", data=grid.numpy())
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"[clip_grid_cache] FAILED {dicom_uuid}: {type(e).__name__}: {e}",
                          flush=True)
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"[clip_grid_cache] {i + 1}/{len(study_ids)} studies "
                      f"(done={done} skipped={skipped} failed={failed}) {elapsed:.0f}s elapsed",
                      flush=True)
                # Periodic flush so a SLURM timeout/OOM/node fault mid-run doesn't corrupt or
                # lose everything written so far -- the resume path (the `dicom_uuid in out_h5`
                # skip above) assumes the file is intact, which only holds if progress is
                # actually flushed to disk before a crash.
                out_h5.flush()

    print(f"[clip_grid_cache] finished: done={done} skipped={skipped} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
