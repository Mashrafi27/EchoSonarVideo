"""Run the echo agentic loop against a vLLM-served model and record what happened.

This is the evaluation counterpart to verl's ToolAgentLoop, deliberately written
standalone rather than reusing it:
  - verl's loop lives inside the RL trainer and needs the whole Ray/FSDP stack up;
    evaluation should be a small job we can debug.
  - we want the TOOL TRACE as a first-class output, not just the final text. The
    agentic metrics (tool call rate, view hit rate) are the point of this project,
    and verl's loop does not hand them back.

It mirrors the rollout contract exactly, so what we measure here is what training
produces: same Hermes <tool_call> format, same composite `echo` tool dispatching on
`op`, same view-menu-of-images opening observation, same image-only tool responses
(INTEGRATION.md §0.2).
"""
import base64
import io
import json
import re

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)

SYSTEM_PROMPT = (
    "You are an expert echocardiographer. You are shown one preview image per "
    "available view of a cardiac ultrasound study.\n"
    "Think step by step inside <think></think>. When you need to look closer, "
    "emit a tool call:\n"
    "<tool_call>\n"
    '{"name": "echo", "arguments": {"op": "select_view", "view_name": "A4C"}}\n'
    "</tool_call>\n"
    "Available ops: select_view (view_name), select_frames (view_name, indices), "
    "zoom (view_name, bbox as [left, top, right, bottom] in PIXELS, frame_indices).\n"
    "When you are ready, give your final answer inside <answer></answer>."
)


def _data_uri(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _image_part(image):
    return {"type": "image_url", "image_url": {"url": _data_uri(image)}}


def parse_tool_calls(text: str) -> list:
    """Extract tool calls from an assistant turn.

    Malformed JSON is DROPPED rather than raised on: a model emitting broken calls
    is a result we want to measure, not a crash. It is still counted, via
    `malformed_tool_calls` in the episode record.
    """
    calls, malformed = [], 0
    for blob in _TOOL_CALL.findall(text or ""):
        try:
            payload = json.loads(blob)
            args = payload.get("arguments") or {}
            if isinstance(args, dict):
                calls.append(args)
            else:
                malformed += 1
        except json.JSONDecodeError:
            malformed += 1
    return calls, malformed


def extract_answer(text: str):
    m = _ANSWER.findall(text or "")
    return m[-1].strip() if m else None


def run_episode(client, model, session, question, overview_frames, *,
                max_turns=6, max_tool_calls=8, max_images=32,
                temperature=0.0, max_tokens=1024):
    """One evaluation episode. Returns a dict, never raises on model behaviour.

    The caps mirror the training-time budget so eval and rollout stay comparable:
    a model that would be cut off during RL must be cut off here too.
    """
    content = [_image_part(f) for f in overview_frames]
    content.append({"type": "text", "text": question})
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]

    trace, images_used = [], len(overview_frames)
    malformed_total, answer, turns_used = 0, None, 0
    finish_reason = "max_turns"

    for turn in range(max_turns):
        turns_used = turn + 1
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens)
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        answer = extract_answer(text)
        if answer is not None:
            finish_reason = "answered"
            break

        calls, malformed = parse_tool_calls(text)
        malformed_total += malformed
        if not calls:
            finish_reason = "no_answer_no_tool"
            break
        if len(trace) >= max_tool_calls:
            finish_reason = "tool_budget"
            break

        obs_content = []
        for args in calls[: max_tool_calls - len(trace)]:
            op = args.get("op")
            try:
                obs = session.run(op, args)
            except Exception as e:                      # I/O errors are data, not crashes
                trace.append({**args, "ok": False, "error": f"{type(e).__name__}: {e}"})
                obs_content.append({"type": "text", "text": f"echo tool error: {e}"})
                continue
            trace.append({**args, "ok": bool(obs.ok),
                          "error": None if obs.ok else obs.error,
                          "n_frames": len(obs.frames) if obs.ok else 0})
            if not obs.ok:
                obs_content.append({"type": "text", "text": obs.error})
                continue
            room = max(0, max_images - images_used)
            frames = obs.frames[:room]
            images_used += len(frames)
            obs_content.extend(_image_part(f.image) for f in frames)
            if obs.text:
                obs_content.append({"type": "text", "text": obs.text})

        if not obs_content:
            obs_content = [{"type": "text", "text": "no observation returned"}]
        messages.append({"role": "user", "content": obs_content})

    return {"answer": answer, "tool_calls": trace, "turns": turns_used,
            "malformed_tool_calls": malformed_total, "finish_reason": finish_reason,
            "images_used": images_used,
            "final_text": messages[-1]["content"] if messages else ""}
