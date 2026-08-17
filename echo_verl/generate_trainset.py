"""Build VeRL v0.7.1 parquet rows for echo RL (hybrid frame path).

Row contract: INTEGRATION.md §0.2. The clip is supplied twice — as the initial
dataset `videos` (true video the model sees up front) and as the tool operand in
extra_info.tools_kwargs.echo.create_kwargs. Row-shaping is unit-tested; the
parquet write lazy-imports pyarrow (UNRUN here — pyarrow lives in the training env).
"""

import json

_DATA_SOURCE = "echo"


def build_row(rl_rec: dict, image_specs: list) -> dict:
    """One row. `image_specs` is the view menu -- one {"image": path} per view.

    NOT a video: Qwen3-VL's video processor resamples a 19-frame list down to 4
    (do_sample_frames=True, fps=2), which would hide most of the menu the agent is
    supposed to choose from. Identical shape to the SFT user turn built by
    echo_rl.sft.serialize -- cold start and rollout must see the same opening context.
    """
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [{"role": "user",
                    "content": "<image>" * len(image_specs) + "\n" + rl_rec["question"]}],
        "videos": [],
        "images": list(image_specs),
        "reward_model": {"ground_truth": json.dumps(rl_rec["reward_key"]), "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {
            "index": rl_rec["study_uuid"],
            "question_type": rl_rec["question_type"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"echo": {"create_kwargs": {"study_uuid": rl_rec["study_uuid"]}}},
        },
    }


def overview_image_specs(rl_rec: dict) -> list:
    """The view menu: one {"image": path} per view.

    verl's process_image -> qwen_vl_utils.fetch_image requires DICTS; a bare path
    string raises TypeError inside fetch_image.
    """
    return [{"image": v["frame"]} for v in rl_rec["overview"]["views"]]


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
            rows.append(build_row(rec, overview_image_specs(rec)))

    write_parquet(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
