"""Build VeRL v0.7.1 parquet rows for echo RL (hybrid frame path).

Row contract: INTEGRATION.md §0.2. The clip is supplied twice — as the initial
dataset `videos` (true video the model sees up front) and as the tool operand in
extra_info.tools_kwargs.echo.create_kwargs. Row-shaping is unit-tested; the
parquet write lazy-imports pyarrow (UNRUN here — pyarrow lives in the training env).
"""

import json

_DATA_SOURCE = "echo"

# Without a cold-start SFT the base model has never seen the <think>/<answer>
# convention, so from a bare question it just tool-calls until the turn budget
# runs out and never emits an answer -> the outcome score is always 0 and GRPO
# gets a flat reward (observed on the first LoRA smoke: reward == 0.1 for every
# rollout). This instruction turn gives the policy the format to imitate.
_SYSTEM = (
    "You are an expert cardiologist reviewing a multi-view echocardiography study. "
    "The images are one preview frame per available view. Use the `echo` tool to "
    "look closer: op=select_view shows preview frames of a view, op=select_frames "
    "returns high-resolution frames, op=zoom crops a region. Reason step by step "
    "inside <think> </think>. Inspect the views you need, then give your final "
    "answer inside <answer> </answer>. Keep the answer concise and clinically "
    "precise, in the same style a report would use."
)


def build_row(rl_rec: dict, image_specs: list) -> dict:
    """One row. `image_specs` is the view menu -- one {"image": path} per view.

    NOT a video: Qwen3-VL's video processor resamples a 19-frame list down to 4
    (do_sample_frames=True, fps=2), which would hide most of the menu the agent is
    supposed to choose from. Identical shape to the SFT user turn built by
    data_core.sft.serialize -- cold start and rollout must see the same opening context.
    """
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [{"role": "system", "content": _SYSTEM},
                   {"role": "user",
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


# The overview strip is a THUMBNAIL menu -- one small frame per view, just enough
# to choose which view to inspect. The frames on disk are native resolution
# (~600-800 px); left unbounded, Qwen3-VL's processor expands 19 of them to
# ~13k vision tokens, blowing past max_prompt_length and making the per-sample
# training tensors ragged (DataProto.concat: "size 13743 vs 13739"). Cap each to
# roughly tool_env's preview_max_side (160 px).
_OVERVIEW_MAX_PIXELS = 180 * 180


def overview_image_specs(rl_rec: dict) -> list:
    """The view menu: one {"image": path, "max_pixels": ...} per view.

    verl's process_image -> qwen_vl_utils.fetch_image requires DICTS; a bare path
    string raises TypeError inside fetch_image. fetch_image honors `max_pixels`
    and downscales to fit.
    """
    return [{"image": v["frame"], "max_pixels": _OVERVIEW_MAX_PIXELS}
            for v in rl_rec["overview"]["views"]]


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
