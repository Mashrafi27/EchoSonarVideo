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
  - initial obs  = the view menu, ONE IMAGE PER VIEW  -> `images`
  - tool obs     = images, one `<image>` per returned frame -> `images`
Nothing emits `videos` any more: Qwen3-VL's video processor resamples a 19-frame
list down to 4, which hid most of the view menu (measured 2026-08-17).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_rl.sft.serialize import serialize_sft   # noqa: E402
from echo_verl.sample import (build_manifest, parse_caps,  # noqa: E402
                              select, write_ids)


def build_sft_row(sft_rec: dict) -> dict:
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
                raise ValueError("video content is no longer emitted; see serialize.py")
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: --per-type structure_description=1 "
               "--per-type abnormality_classification=1  (keeps the three "
               "one-per-study report types whole, caps the two bulk types)")
    ap.add_argument("--sft-jsonl", default="build/sft.jsonl")
    ap.add_argument("--out", default="build/sft_train.parquet")
    ap.add_argument("--per-type", action="append", default=[], metavar="TYPE=N",
                    help="max records of TYPE per study; repeatable. "
                         "Types not named are kept whole unless --default-per-type.")
    ap.add_argument("--default-per-type", type=int, default=None,
                    help="cap for types not named by --per-type (default: keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="where to write the sampling manifest "
                         "(default: <out>.manifest.json)")
    ap.add_argument("--limit", type=int, default=None,
                    help="DEBUG ONLY: truncate to the first N rows of the input. "
                         "Not random and not stratified; use --per-type for real "
                         "runs. A forgotten --limit once trained a full run on "
                         "2.3%% of the corpus.")
    args = ap.parse_args(argv)

    records = list(iter_jsonl(args.sft_jsonl))
    n_source = len(records)

    if args.limit is not None:
        print(f"WARNING: --limit {args.limit} takes the FIRST {args.limit} records, "
              f"not a sample. Debug only.")
        selected = records[: args.limit]
    else:
        selected = select(records, caps=parse_caps(args.per_type),
                          default_cap=args.default_per_type, seed=args.seed)

    out = Path(args.out)
    ids_path = out.with_suffix(".ids.txt")
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")

    rows = [build_sft_row(rec) for rec in selected]
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), str(out))

    write_ids(ids_path, selected)
    manifest = build_manifest(
        selected=selected, source=args.sft_jsonl,
        caps=parse_caps(args.per_type), default_cap=args.default_per_type,
        seed=args.seed, out_path=out, ids_path=ids_path, n_source_records=n_source)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")

    sel = manifest["selected"]
    print(f"wrote {len(rows)} rows ({sel['n_studies']} studies) -> {out}")
    for qtype, n in sel["by_question_type"].items():
        print(f"  {qtype:32s} {n:7,}")
    print(f"manifest -> {manifest_path}  (ids_sha256 {sel['ids_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
