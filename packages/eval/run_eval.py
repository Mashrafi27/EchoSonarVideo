"""Evaluate a served model on the held-out echo test set.

    # served vLLM
    python -m eval.run_eval --base-url http://NODE:8000/v1 \
        --model echo --limit 200 --out build/eval_step100.jsonl

    # local HF generate (one process per GPU, set HIP_VISIBLE_DEVICES externally)
    python -m eval.run_eval --local-model /path/to/model \
        --out build/eval_base.jsonl --prompt-mode plain

Writes one JSON line per episode (answer, tool trace, finish reason) and prints
nothing but progress -- scoring is a separate step (score_eval.py) so a slow
generation run is never repeated to change a metric.
"""
import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_env.config import EnvConfig                       # noqa: E402
from eval.agentic_loop import (run_episode,        # noqa: E402
                                        run_plain_episode)
from verl_bridge.session import EchoSession                   # noqa: E402


# ---------------------------------------------------------------------------
# Local HF-generate client (used when --local-model is given instead of --base-url)
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, text): self.content = text

class _Choice:
    def __init__(self, text): self.message = _Msg(text)

class _Resp:
    def __init__(self, text): self.choices = [_Choice(text)]

class _Completions:
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def create(self, model, messages, temperature=0.0, max_tokens=4096, **_):
        import torch
        from PIL import Image

        hf_messages, images = [], []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                hf_messages.append({"role": msg["role"], "content": content})
                continue
            hf_content = []
            for part in content:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    raw = base64.b64decode(url.split(",", 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    hf_content.append({"type": "image"})
                elif part["type"] == "text":
                    hf_content.append({"type": "text", "text": part["text"]})
            hf_messages.append({"role": msg["role"], "content": hf_content})

        prompt = self._processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[prompt],
            images=images or None,
            return_tensors="pt",
            do_sample_frames=False,
        ).to(self._model.device)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )
        n_input = inputs["input_ids"].shape[1]
        text = self._processor.tokenizer.decode(
            out_ids[0, n_input:], skip_special_tokens=True)
        return _Resp(text)


class _Chat:
    def __init__(self, model, processor):
        self.completions = _Completions(model, processor)


class LocalModelClient:
    """Drop-in for openai.OpenAI that runs Qwen3-VL locally via HF generate."""
    def __init__(self, model_path):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        print(f"[eval] loading {model_path} ...", flush=True)
        self._processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="cuda", local_files_only=True)
        self._model.eval()
        self.chat = _Chat(self._model, self._processor)
        print("[eval] model ready", flush=True)


def load_records(path, limit=None, per_type=None, seed=0):
    """Sample eval records. `per_type` caps each question type so a 200-episode
    run is not 88% short-answer questions, matching the corpus imbalance."""
    records = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    if per_type:
        kept, seen = [], {}
        for r in records:
            q = r["question_type"]
            if seen.get(q, 0) >= per_type:
                continue
            seen[q] = seen.get(q, 0) + 1
            kept.append(r)
        records = kept
    return records[:limit] if limit else records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-url",
                      help="OpenAI-compatible server (a real served vLLM engine).")
    mode.add_argument("--local-model",
                      help="HF model path; loads directly via transformers.generate, "
                           "one process per GPU (set HIP_VISIBLE_DEVICES externally).")
    ap.add_argument("--model", default="echo", help="model name sent to the server")
    ap.add_argument("--eval-jsonl", default="build/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-type", type=int, default=None,
                    help="cap episodes per question type (balances the mix)")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=64,
                    help="total image budget per episode, OVERVIEW INCLUDED. "
                         "Studies carry a median of 18 views (max 19), so the "
                         "view menu alone consumed 18 of the old 32 and left "
                         "room for 2-3 tool calls -- episodes died on the image "
                         "cap, not on model behaviour (smoke job 144199). 64 "
                         "makes max-tool-calls the binding constraint again.")
    ap.add_argument("--prompt-mode", choices=("agentic", "plain"), default="agentic",
                    help="'plain' is a single non-agentic turn with no tools -- how "
                         "EchoSonar-R evaluated an untrained base model. Under the "
                         "agentic prompt a base model scores ~0 for format reasons.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split records across N processes; each takes i::N "
                         "AFTER sampling, so shards partition one fixed episode set")
    args = ap.parse_args(argv)

    if args.local_model:
        client = LocalModelClient(args.local_model)
    else:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    cfg = EnvConfig.from_env()

    records = load_records(args.eval_jsonl, args.limit, args.per_type, args.seed)
    # Shard AFTER sampling: every shard count/mix is then a deterministic slice of
    # the same episode set, so concatenating shards reproduces the unsharded run.
    if args.num_shards > 1:
        records = records[args.shard::args.num_shards]
    print(f"[eval] shard {args.shard}/{args.num_shards}: {len(records)} episodes "
          f"({args.prompt_mode} prompt) -> {args.out}", flush=True)

    written = 0
    with open(args.out, "w") as w:
        for i, rec in enumerate(records):
            session = EchoSession(cfg, rec["study_uuid"])
            views = rec["overview"]["views"]
            frames = [session.loader.load(v["frame"]) for v in views]
            try:
                if args.prompt_mode == "plain":
                    ep = run_plain_episode(client, args.model, session,
                                           rec["question"], frames,
                                           max_images=args.max_images,
                                           temperature=args.temperature)
                else:
                    ep = run_episode(client, args.model, session, rec["question"], frames,
                                     max_turns=args.max_turns,
                                     max_tool_calls=args.max_tool_calls,
                                     max_images=args.max_images,
                                     temperature=args.temperature)
            except Exception as e:            # a dead server should not lose prior work
                print(f"[eval] episode {i} FAILED: {type(e).__name__}: {e}", flush=True)
                ep = {"answer": None, "tool_calls": [], "turns": 0,
                      "malformed_tool_calls": 0, "finish_reason": "error",
                      "images_used": 0, "error": f"{type(e).__name__}: {e}"}
            w.write(json.dumps({
                "study_uuid": rec["study_uuid"],
                "question_type": rec["question_type"],
                "question": rec["question"],
                "gold_answer": rec["answer"],
                "reward_key": rec["reward_key"],
                "views_available": [v["view"] for v in views],
                **ep}) + "\n")
            w.flush()
            written += 1
            if written % 25 == 0:
                print(f"[eval] {written}/{len(records)}", flush=True)
    print(f"[eval] wrote {written} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
