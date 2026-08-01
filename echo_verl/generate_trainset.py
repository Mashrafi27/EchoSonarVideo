"""Build VeRL v0.7.1 parquet rows for echo RL (hybrid frame path).

Row contract: INTEGRATION.md §0.2. The clip is supplied twice — as the initial
dataset `videos` (true video the model sees up front) and as the tool operand in
extra_info.tools_kwargs.echo.create_kwargs. Row-shaping is unit-tested; the
parquet write lazy-imports pyarrow (UNRUN here — pyarrow lives in the training env).
"""

_DATA_SOURCE = "echo"


def build_row(rl_rec: dict, video_spec: dict) -> dict:
    return {
        "data_source": _DATA_SOURCE,
        "agent_name": "tool_agent",
        "prompt": [{"role": "user", "content": "<video>\n" + rl_rec["question"]}],
        "videos": [video_spec],
        "images": [],
        "reward_model": {"ground_truth": rl_rec["reward_key"], "style": "rule"},
        "ability": "echo_vqa",
        "extra_info": {
            "index": rl_rec["study_uuid"],
            "question_type": rl_rec["question_type"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"echo": {"create_kwargs": {"study_uuid": rl_rec["study_uuid"]}}},
        },
    }


def write_parquet(rows: list, path: str) -> int:
    import pyarrow as pa            # lazy: not installed in the offline .venv
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)
