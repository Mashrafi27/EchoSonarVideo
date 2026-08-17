"""Build the cold-start SFT parquet for verl's multi-turn SFT trainer.

Input : build/sft.jsonl (Phase-1 records: study_uuid, question, trajectory)
Output: parquet with verl's MultiTurnSFTDataset columns —
        `messages` (string content carrying <video>/<image> placeholders),
        `videos`, `images`.

verl's `MultiTurnSFTDataset._build_messages` splits STRING content on the literal
`<video>`/`<image>` tokens and asserts the placeholder count equals the length of the
corresponding column, so the canonical message list from `echo_rl.sft.serialize` (which
carries structured content) is flattened here.

Frame path is HYBRID (INTEGRATION.md §0.2), and the shapes mirror RL exactly:
  - initial obs  = ONE video over the per-view overview thumbnails  -> `videos[0]`
  - tool obs     = images, one `<image>` per returned frame         -> `images`
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_rl.sft.serialize import serialize_sft   # noqa: E402


def build_sft_row(sft_rec: dict, *, fps: float = 1.0, max_frames: int | None = None) -> dict:
    """Serializer messages -> one verl SFT row (placeholders + media columns)."""
    messages = serialize_sft(sft_rec["trajectory"], sft_rec["question"])
    out_messages, videos, images = [], [], []

    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):                      # assistant turns
            out_messages.append({"role": msg["role"], "content": content})
            continue

        parts = []
        for item in content:
            if item["type"] == "video":
                spec = {"video": list(item["frames"]), "fps": fps}
                if max_frames is not None:
                    spec["max_frames"] = max_frames
                videos.append(spec)
                parts.append("<video>")
            elif item["type"] == "image":
                # verl's process_image -> qwen_vl_utils.fetch_image wants a DICT
                # ({"image": path}); a bare path string raises TypeError inside
                # fetch_image. Videos take a bare list of paths, images do not.
                images.extend({"image": f} for f in item["frames"])
                parts.append("<image>" * len(item["frames"]))
            else:
                parts.append(item["text"])
        out_messages.append({"role": msg["role"], "content": "\n".join(parts)})

    return {"messages": out_messages, "videos": videos, "images": images}


def iter_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-jsonl", default="build/sft.jsonl")
    ap.add_argument("--out", default="build/sft_train.parquet")
    ap.add_argument("--limit", type=int, default=None, help="write only the first N rows")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    rows = []
    for i, rec in enumerate(iter_jsonl(args.sft_jsonl)):
        if args.limit is not None and i >= args.limit:
            break
        rows.append(build_sft_row(rec, fps=args.fps, max_frames=args.max_frames))

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
