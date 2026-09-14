"""Build the frozen-encoder embedding cache, once per study, reused across every QA pair.

The encoder is frozen -- its output for a study never changes regardless of which of that
study's ~25 QA pairs is being trained on, so running it once per study (not once per QA pair,
128,215 forward passes) is both correct and the whole point of caching. Same shape as darya's
own echoprime_study_cache.pt (found on this box) for the EchoPrime TEXT tower -- this is the
video-tower equivalent, our own studies.

Usage:
    python -m echoprime_track.build_video_cache --studies-dir "$ECHO_PREPROCESSED_DIR" \\
        --out-dir build/echoprime_video_cache

Idempotent / resumable: skips a study if its cache file already exists, so a killed run can
just be relaunched.
"""
import argparse
import os
import time

import torch

from echoprime_track.encoder import embed_videos, load_frozen_encoder
from echoprime_track.preprocess import study_to_tensor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies-dir", required=True,
                     help="preprocessed tree root, one subdir per study_uuid")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--study-list", default=None,
                     help="optional file, one study_uuid per line, to restrict/order the run "
                          "(default: every subdir of --studies-dir)")
    ap.add_argument("--bin-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.study_list:
        with open(args.study_list) as fh:
            study_ids = [l.strip() for l in fh if l.strip()]
    else:
        study_ids = sorted(
            d for d in os.listdir(args.studies_dir)
            if os.path.isdir(os.path.join(args.studies_dir, d)))
    if args.limit:
        study_ids = study_ids[:args.limit]

    print(f"[echoprime_track] {len(study_ids)} studies queued -> {args.out_dir}", flush=True)
    encoder = load_frozen_encoder()

    done = skipped = failed = 0
    t0 = time.monotonic()
    for i, study_uuid in enumerate(study_ids):
        out_path = os.path.join(args.out_dir, f"{study_uuid}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        study_dir = os.path.join(args.studies_dir, study_uuid)
        try:
            stack, view_names = study_to_tensor(study_dir)
            embeddings = embed_videos(encoder, stack, bin_size=args.bin_size)
            torch.save({"view_names": view_names, "embeddings": embeddings}, out_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"[echoprime_track] FAILED {study_uuid}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            print(f"[echoprime_track] {i + 1}/{len(study_ids)} "
                  f"(done={done} skipped={skipped} failed={failed}) "
                  f"{elapsed:.0f}s elapsed", flush=True)

    print(f"[echoprime_track] finished: done={done} skipped={skipped} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
