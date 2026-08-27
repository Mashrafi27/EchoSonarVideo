"""Evaluate a served model on the held-out echo test set.

    python -m echo_verl.eval.run_eval --base-url http://NODE:8000/v1 \
        --model echo --limit 200 --out build/eval_step100.jsonl

Writes one JSON line per episode (answer, tool trace, finish reason) and prints
nothing but progress -- scoring is a separate step (score_eval.py) so a slow
generation run is never repeated to change a metric.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from echo_env.config import EnvConfig                       # noqa: E402
from echo_verl.eval.agentic_loop import (run_episode,        # noqa: E402
                                        run_plain_episode)
from echo_verl.session import EchoSession                   # noqa: E402


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
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible server. Omit to run the model in-process.")
    ap.add_argument("--local-model", default=None,
                    help="path to a HF model dir; runs in-process via LocalHFClient")
    ap.add_argument("--model", default="echo",
                    help="model name sent to the server (ignored in local mode)")
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

    if bool(args.base_url) == bool(args.local_model):
        ap.error("give exactly one of --base-url or --local-model")
    if args.local_model:
        # In-process because the installed vLLM is a CUDA wheel and cannot serve
        # on MI210; see echo_verl/eval/local_client.py for why no container works.
        from echo_verl.eval.local_client import LocalHFClient
        client = LocalHFClient(args.local_model)
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
