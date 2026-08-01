"""Serialize a Phase-1 trajectory dict into a Qwen3-VL multi-turn messages list.

Tool-call convention = composite EchoTool op-dispatch (INTEGRATION.md §0.1):
    <tool_call>
    {"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}
    </tool_call>
The exact chat-template/role wrapping into verl's SFT trainer is a P3c concern;
this produces a canonical messages list only.
"""
import json

_OPENING = "To answer this I should examine the relevant views."


def _tool_call_json(name: str, args: dict) -> str:
    # Phase-1 trajectory uses args={"view": ...}; map to the op-dispatch schema.
    op_args = {"op": name, "view_name": args["view"]}
    payload = {"name": "echo", "arguments": op_args}
    return "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"


def _assistant(think: str, tail: str) -> dict:
    return {"role": "assistant", "content": f"<think>{think}</think>\n{tail}"}


def _iter_tool_think(turns: list):
    """Yield (tool_turn, findings_text) pairs from the flat [tool, think, ...] list."""
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.get("type") != "tool":
            i += 1
            continue
        think = ""
        if i + 1 < len(turns) and turns[i + 1].get("type") == "think":
            think = turns[i + 1]["text"]
        yield t, think
        i += 2


def serialize_sft(traj: dict, question: str, *, opening_think: str = _OPENING) -> list[dict]:
    user_content = [{"type": "text", "text": question}]
    for v in traj["overview"]["views"]:
        user_content.append({"type": "video", "frames": [v["frame"]], "view": v["view"]})
    messages = [{"role": "user", "content": user_content}]

    pending_think = opening_think
    for tool, findings in _iter_tool_think(traj["turns"]):
        messages.append(_assistant(pending_think, _tool_call_json(tool["name"], tool["args"])))
        obs = tool["obs"]
        messages.append({"role": "tool",
                         "content": [{"type": "video", "frames": obs["frames"], "view": obs["view"]}]})
        pending_think = findings

    messages.append(_assistant(pending_think, f"<answer>{traj['answer']}</answer>"))
    return messages
