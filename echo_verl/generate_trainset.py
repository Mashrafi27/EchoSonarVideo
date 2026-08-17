"""Build VeRL v0.7.1 parquet rows for echo RL (hybrid frame path).

Row contract: INTEGRATION.md §0.2. The clip is supplied twice — as the initial
dataset `videos` (true video the model sees up front) and as the tool operand in
extra_info.tools_kwargs.echo.create_kwargs. Row-shaping is unit-tested; the
parquet write lazy-imports pyarrow (UNRUN here — pyarrow lives in the training env).
"""

import json

_DATA_SOURCE = "echo"


def build_row(rl_rec: dict, video_spec: dict) -> dict:
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [{"role": "user", "content": "<video>\n" + rl_rec["question"]}],
        "videos": [video_spec],
        "images": [],
        "reward_model": {"ground_truth": json.dumps(rl_rec["reward_key"]), "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {
            "index": rl_rec["study_uuid"],
            "question_type": rl_rec["question_type"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"echo": {"create_kwargs": {"study_uuid": rl_rec["study_uuid"]}}},
        },
    }


def overview_video_spec(rl_rec: dict, *, fps: float = 1.0, max_frames=None) -> dict:
    """Initial observation = ONE video over the per-view overview thumbnails.

    The agent's job is choosing views, so the opening context must be the view menu;
    a single default-view clip would make `select_view` unlearnable. Kept identical to
    the SFT user turn (echo_rl.sft.serialize) so cold start and rollout agree — see
    INTEGRATION.md §0.2.
    """
    spec = {"video": [v["frame"] for v in rl_rec["overview"]["views"]], "fps": fps}
    if max_frames is not None:
        spec["max_frames"] = max_frames
    return spec


def write_parquet(rows: list, path: str) -> int:
    import pyarrow as pa            # lazy: not installed in the offline .venv
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)


def main(argv=None) -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Build the verl RL parquet from build/rl.jsonl")
    ap.add_argument("--rl-jsonl", default="build/rl.jsonl")
    ap.add_argument("--out", default="build/rl_train.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    rows = []
    with open(args.rl_jsonl) as fh:
        for i, line in enumerate(fh):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            rows.append(build_row(rec, overview_video_spec(
                rec, fps=args.fps, max_frames=args.max_frames)))

    write_parquet(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
