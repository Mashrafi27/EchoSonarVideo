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
    # Initial observation = the view menu: ONE IMAGE PER VIEW, not a video.
    #
    # It was a single video over the thumbnails until 2026-08-17, when measurement
    # showed Qwen3-VL's video processor (do_sample_frames=True, fps=2, min_frames=4)
    # treats a 19-frame list as a ~0.8 s clip and resamples it down to FOUR frames:
    # video_grid_thw [[2, 24, 24]]. The agent was being asked to select among 19 views
    # while seeing 4 of them. A view menu is not a temporal sequence, so images are
    # both the correct semantics and immune to every frame-sampling knob.
    # echo_verl.generate_trainset.build_row emits the identical shape for RL.
    views = traj["overview"]["views"]
    user_content = [{"type": "image", "frames": [v["frame"]], "views": [v["view"]]}
                    for v in views]
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]

    pending_think = opening_think
    for tool, findings in _iter_tool_think(traj["turns"]):
        messages.append(_assistant(pending_think, _tool_call_json(tool["name"], tool["args"])))
        obs = tool["obs"]
        # HYBRID frame path (INTEGRATION.md §0.2): tool observations come back as
        # IMAGES at RL time (ToolResponse(image=[...]) -- v0.7.1's ToolAgentLoop
        # refuses tool-returned video). SFT must teach the same message shape the
        # rollout produces, so tool obs are tagged "image", not "video".
        messages.append({"role": "tool",
                         "content": [{"type": "image", "frames": obs["frames"], "view": obs["view"]}]})
        pending_think = findings

    messages.append(_assistant(pending_think, f"<answer>{traj['answer']}</answer>"))
    return messages
