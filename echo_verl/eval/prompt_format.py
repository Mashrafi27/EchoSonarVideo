"""Convert OpenAI-style chat messages into the Qwen chat-template shape.

Used by `scripts/check_prompt_parity.py` to confirm the harness renders the same
prompt a served vLLM engine would (vLLM's /v1/chat/completions applies the
processor's chat template with add_generation_prompt=True and turns `image_url`
parts into the template's image placeholders; this does the same, via the SAME
processor, so the two can be compared token-for-token).
"""
from __future__ import annotations

import base64
import io

from PIL import Image


def _decode_part(part):
    """OpenAI content part -> ('image', PIL) | ('text', str) | None."""
    kind = part.get("type")
    if kind == "text":
        return ("text", part.get("text", ""))
    if kind == "image_url":
        url = part["image_url"]["url"]
        if not url.startswith("data:"):
            raise ValueError("only accepts inline data: image URLs")
        raw = base64.b64decode(url.split(",", 1)[1])
        return ("image", Image.open(io.BytesIO(raw)).convert("RGB"))
    return None


def to_qwen_messages(messages):
    """OpenAI-style messages -> (chat-template messages, flat list of PIL images).

    Images are replaced by bare {"type": "image"} placeholders in template order and
    collected into one list, which is how the Qwen3-VL processor expects them.
    """
    out, images = [], []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for part in content or []:
            decoded = _decode_part(part)
            if decoded is None:
                continue
            kind, value = decoded
            if kind == "image":
                images.append(value)
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": value})
        out.append({"role": msg["role"], "content": parts})
    return out, images
