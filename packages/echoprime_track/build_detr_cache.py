"""Build our own RT-DETR detection h5 cache, replacing Darya's {train,test}_detections*.h5.
Same per-dicom-per-frame schema (darya_cache.py's DETR readers consume it unchanged) -- see
docs/superpowers/specs/2026-09-16-echoprime-self-inference-design.md.

Frame selection MUST match preprocess.clip_to_tensor's (native indices 0,2,4,...,30 -- 16
frames from up to 32 native) so this cache's frame_N positions line up with the clip grid's
temporal groups grid.py indexes into. A short clip's out-of-range positions are simply not
written (no frame_N key), same convention darya_cache.py already handles for "no detections".
"""
import argparse
import glob
import os
import time

import h5py

from data_core.data.views import parse_clip_dirname
from echoprime_track.detector import detect_frame, load_detector
from echoprime_track.preprocess import FRAME_STRIDE, FRAMES_TO_TAKE, load_clip_frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--study-list", default=None)
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

    print(f"[detr_cache] {len(study_ids)} studies queued -> {args.out}", flush=True)
    model = load_detector()

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
                    frames = load_clip_frames(clip_dir, max_frames=FRAMES_TO_TAKE)
                    # Stage frame results in memory first, only write to h5 after loop completes
                    frame_results = {}  # {pos: det_dict}
                    for pos, native_idx in enumerate(range(0, FRAMES_TO_TAKE, FRAME_STRIDE)):
                        if native_idx >= len(frames):
                            continue  # short clip -- matches clip_to_tensor's zero-pad position
                        det = detect_frame(model, frames[native_idx])
                        if det["classes"].shape[0] == 0:
                            continue  # nothing detected -- no frame_N group, same as missing
                        frame_results[pos] = det
                    # All frames processed successfully -- now write to h5
                    grp = out_h5.create_group(dicom_uuid)
                    for pos, det in frame_results.items():
                        fg = grp.create_group(f"frame_{pos}")
                        fg.create_dataset("boxes_xyxy", data=det["boxes_xyxy"])
                        fg.create_dataset("classes", data=det["classes"])
                        fg.create_dataset("confidences", data=det["confidences"])
                        fg.create_dataset("embeddings", data=det["embeddings"])
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"[detr_cache] FAILED {dicom_uuid}: {type(e).__name__}: {e}", flush=True)
            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"[detr_cache] {i + 1}/{len(study_ids)} studies "
                      f"(done={done} skipped={skipped} failed={failed}) {elapsed:.0f}s elapsed",
                      flush=True)
                # Periodic flush so a SLURM timeout/OOM/node fault mid-run doesn't corrupt or
                # lose everything written so far -- the resume path (the dicom_uuid-in-out_h5
                # skip above) assumes the file is intact, which only holds if progress is
                # actually flushed to disk before a crash.
                out_h5.flush()

    print(f"[detr_cache] finished: done={done} skipped={skipped} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
