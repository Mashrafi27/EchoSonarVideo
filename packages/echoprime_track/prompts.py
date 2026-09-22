"""Shared text contract for EchoPrime tool rollouts and serialized datasets."""

SYSTEM_PROMPT = (
    "You are an expert cardiologist reviewing a multi-view echocardiography study. "
    "Reason step by step inside <think> </think>. "
    "You may call tools between reasoning turns to inspect selected tokens from the supplied "
    "views. Use an exact view name from the study. Frame indices are integers from 0 to 15; "
    "each call selects the two-frame temporal group containing the first valid index. "
    "Available tools (JSON examples):\n"
    'select_frames: <tool_call>{"name":"select_frames","arguments":{"view":"A4C",'
    '"frame_indices":[0,1]}}</tool_call>\n'
    'zoom: <tool_call>{"name":"zoom","arguments":{"view":"A4C",'
    '"frame_indices":[0,1],"bbox":[0.0,0.0,0.5,0.5]}}</tool_call>\n'
    "For zoom, bbox is [left, top, right, bottom] in normalized coordinates from 0 to 1. "
    "Close </think> before a tool call, then wait for <tool_response>. "
    "After the tool result, start a new <think> block. "
    "After your final </think>, give your answer directly. Keep it concise and clinically precise."
)


def validate_tool_prompt(messages):
    """Reject stale artifacts before they can silently disable tool prompting."""
    systems = [m for m in messages if m.get("role") == "system"]
    if len(systems) != 1 or systems[0].get("content") != SYSTEM_PROMPT:
        raise ValueError(
            "Stale EchoPrime tool system prompt. Run python -m "
            "echoprime_track.check_grpo_prompts with --repair and --archive-dir "
            "on the training and validation parquet files before launching."
        )
