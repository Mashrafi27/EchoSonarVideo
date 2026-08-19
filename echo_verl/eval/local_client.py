"""An in-process stand-in for an OpenAI client, backed by HuggingFace generate().

WHY THIS EXISTS
---------------
`agentic_loop.run_episode` talks to a served model through exactly one call:

    client.chat.completions.create(model=, messages=, temperature=, max_tokens=)
        -> resp.choices[0].message.content

That is the entire contract, so anything implementing it can drive the loop. The
vLLM installed on this cluster is a CUDA wheel (`_C.abi3.so` links libcudart.so.12,
no `_rocm_C`) and cannot serve on MI210; every container route to a ROCm build is
closed on this filesystem (VAST rejects ':' in filenames, no docker group, no
writable non-VAST scratch, no OCI-SIF support in apptainer 1.4.2 here). Rather than
block evaluation on that, we run the model in-process.

This is slower than a served engine -- there is no continuous batching and every
turn re-prefills the whole conversation -- but evaluation is a few thousand short
episodes, it shards trivially across the 8 GPUs of one node, and it removes a
server from the debugging surface. GRPO still needs a real vLLM; evaluation does not.

FIDELITY
--------
The point is to measure the MODEL, not the harness, so the prompt this renders must
match what a vLLM server would render. vLLM's /v1/chat/completions applies the
processor's chat template with add_generation_prompt=True and turns `image_url`
parts into the template's image placeholders; so do we, via the SAME processor.
`scripts/check_prompt_parity.py` asserts the token ids agree.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import torch
from PIL import Image


def _decode_part(part):
    """OpenAI content part -> ('image', PIL) | ('text', str) | None."""
    kind = part.get("type")
    if kind == "text":
        return ("text", part.get("text", ""))
    if kind == "image_url":
        url = part["image_url"]["url"]
        if not url.startswith("data:"):
            raise ValueError("local client only accepts inline data: image URLs")
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


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, *, model=None, messages, temperature=0.0, max_tokens=1024, **_):
        return self._owner._generate(messages, temperature, max_tokens)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class LocalHFClient:
    """Drop-in for `openai.OpenAI` covering only chat.completions.create."""

    def __init__(self, model_path, *, device="cuda", dtype=torch.bfloat16,
                 attn_implementation="sdpa"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_implementation,
        ).to(device).eval()
        self.device = device
        self.chat = _Chat(self)

    def build_inputs(self, messages):
        """Render one conversation exactly as a server would. Exposed for the parity test."""
        qwen_messages, images = to_qwen_messages(messages)
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True)
        # images=None (not []) when there are none: an empty list makes some
        # processors emit a zero-length pixel tensor instead of omitting it.
        return self.processor(text=[text], images=images or None,
                              return_tensors="pt")

    @torch.no_grad()
    def _generate(self, messages, temperature, max_tokens):
        inputs = self.build_inputs(messages).to(self.device)
        # temperature=0 means GREEDY. Passing do_sample=True with temperature=0.0
        # is a division by zero that HF either errors on or silently garbles --
        # vLLM maps it to greedy, so we must too or eval and rollout diverge.
        if temperature and temperature > 0:
            gen = dict(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen = dict(do_sample=False)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                  pad_token_id=self.processor.tokenizer.pad_token_id,
                                  **gen)
        # generate() returns prompt + continuation; keep only what was added.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _Response(choices=[_Choice(message=_Message(content=text))])
