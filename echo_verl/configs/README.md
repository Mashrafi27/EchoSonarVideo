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
- Row contract: `clip` (supplied twice for repeat reliability), `view_name`, `gt_frames`, `question`, `answer`, `rl_record` (tool-use history)

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

Validate the tool config parses before launching:
```bash
python -c "import yaml; d=yaml.safe_load(open('echo_verl/configs/echo_tool_config.yaml')); print(d['tools'][0]['class_name'])"
```
Expected: `echo_verl.echo_tool.EchoTool`
