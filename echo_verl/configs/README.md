# VeRL Launch Wiring for Echo Tool

**STATUS: UNRUN — These are the flags a GPU operator sets; validate end-to-end before RL.**

This document describes the concrete configuration needed to launch VeRL training with the Echo tool.

## Multi-Turn Configuration

The following flags must be set in the VeRL rollout config:

- `rollout.multi_turn.enable=true` — Enable multi-turn dialog with tool use
- `rollout.multi_turn.format=hermes` — Use Hermes format for tool calls
- `rollout.multi_turn.tool_config_path=echo_verl/configs/echo_tool_config.yaml` — Path to tool registry
- `rollout.multi_turn.max_assistant_turns=<N>` — Maximum assistant turns (set by operator)

## Reward Configuration

Custom reward function for echo tool:

- `custom_reward_function.path=echo_verl/reward.py` — Reward module path
- `custom_reward_function.name=compute_score` — Entrypoint function name

## Dataset Configuration

Training data is generated from `echo_verl.generate_trainset`:

- Parquet dataset with rows carrying `agent_name="tool_agent"`
- Row contract (per INTEGRATION.md §0.2, as emitted by `build_row`):
  - `data_source` — `"echo"`
  - `agent_name` — `"tool_agent"` (selects ToolAgentLoop)
  - `prompt` — `[{"role": "user", "content": "<video>\n<question>"}]`
  - `videos` — `[video_spec]`; the clip is supplied twice: once here (initial full-clip
    video the model sees up front) and again as the tool operand in
    `extra_info.tools_kwargs.echo.create_kwargs.study_uuid`
  - `images` — `[]`
  - `reward_model` — `{"ground_truth": <JSON-encoded reward_key string>, "style": "rule"}`.
    `ground_truth` is JSON-encoded (not a raw dict/list) so every row's column has a
    uniform `str` type — required for `pa.Table.from_pylist` to unify the Arrow schema
    across mixed question types (yes/no and text targets are `str`, `abnormality_list`
    targets are `list`). `echo_verl/reward.py::compute_score` parses it back via
    `_parse_reward_key`.
  - `ability` — `"echo_vqa"`
  - `extra_info` — `{"index", "question_type", "need_tools_kwargs": True,
    "tools_kwargs": {"echo": {"create_kwargs": {"study_uuid": ...}}}}`

## Model Configuration

Base model and dependencies:

- Base model: `Qwen3-VL-8B-Instruct`
- Required: `transformers>=4.57`
- Video key in dataset: `data.video_key=videos`

## VeRL GRPO/SFT Launch

These flags wire into the verl-071 rollout config. The GPU operator should:

1. Set the multi-turn flags above
2. Configure reward function path and name
3. Point dataset to parquet output of `echo_verl.generate_trainset`
4. Use transformers >=4.57 with Qwen3-VL-8B-Instruct base
5. Set `data.video_key=videos` to match dataset video column name

Validate the whole training environment (this config included) before launching:
```bash
python scripts/check_train_env.py
```
See `docs/TRAINING_ENV.md` for the pinned versions, install order, and what each
check asserts. The schema round-trip (tool name `echo`, `op` enum survives
pydantic's `extra="ignore"`) also runs offline: `pytest echo_verl/tests/test_tool_config.py`.
