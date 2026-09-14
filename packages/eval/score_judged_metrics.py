"""Compute the metrics that need a model/judge -- BERTScore, GREEN (ours,
public prompt), and Table 2 reasoning quality (ours, from EchoSonar-R's
published rubric definitions) -- as a follow-up pass over an episodes.jsonl
from run_eval.py. Kept separate from score_eval.py/score_per_disease.py so
those stay import-light (no torch/bert_score, no judge server needed).

    # BERTScore only (local model, no server):
    python -m eval.score_judged_metrics build/eval_x.jsonl --bertscore

    # + GREEN-approx and Table 2 reasoning quality, judged by a served model:
    python -m eval.score_judged_metrics build/eval_x.jsonl \\
        --bertscore --green --reasoning-quality \\
        --judge-base-url http://localhost:8001/v1 --judge-model mistral-7b

Every GREEN/reasoning-quality number this prints is explicitly labeled "ours"
-- see green_approx.py and reasoning_quality.py docstrings for why they are
not EchoSonar-R's Table 2/3 numbers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.agentic_loop import extract_think            # noqa: E402
from eval.score_eval import resolve_answer             # noqa: E402

REPORT_TYPE = "full_report"    # matches EchoSonar-R Table 3's task


def _report_episodes(episodes):
    return [e for e in episodes if e.get("question_type") == REPORT_TYPE]


def run_bertscore(episodes):
    from eval.bertscore import bertscore_corpus
    eps = _report_episodes(episodes)
    preds = [resolve_answer(e) for e in eps]
    refs = [e.get("gold_answer") or "" for e in eps]
    return bertscore_corpus(preds, refs)


def run_green(episodes, client, model):
    from eval.green_approx import green_score_corpus
    eps = _report_episodes(episodes)
    preds = [resolve_answer(e) for e in eps]
    refs = [e.get("gold_answer") or "" for e in eps]
    return green_score_corpus(client, model, preds, refs)


def run_reasoning_quality(episodes, client, model):
    from eval.reasoning_quality import reasoning_quality_corpus
    records = []
    for e in episodes:
        trace = extract_think(e.get("final_text"))
        if not trace:
            # base / non-SFT models don't use <think> tags -- fall back to the
            # full response, which IS the reasoning+answer to judge in that case.
            trace = e.get("output_text") or resolve_answer(e) or ""
        if not trace:
            continue
        records.append({"question": e.get("question"), "reasoning_trace": trace,
                        "final_answer": resolve_answer(e)})
    return reasoning_quality_corpus(client, model, records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes")
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--green", action="store_true",
                    help="GREEN (ours, public prompt) -- needs --judge-base-url/--judge-model")
    ap.add_argument("--reasoning-quality", action="store_true",
                    help="Table 2 (ours, from published rubric) -- needs --judge-base-url/--judge-model")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if (args.green or args.reasoning_quality) and not (args.judge_base_url and args.judge_model):
        ap.error("--green/--reasoning-quality need --judge-base-url and --judge-model")

    episodes = [json.loads(l) for l in open(args.episodes) if l.strip()]
    out = {}

    if args.bertscore:
        print("[score_judged_metrics] running BERTScore (local PubMedBERT)...", file=sys.stderr)
        out["bertscore"] = run_bertscore(episodes)

    client = None
    if args.green or args.reasoning_quality:
        from openai import OpenAI
        client = OpenAI(base_url=args.judge_base_url, api_key="EMPTY")

    if args.green:
        print("[score_judged_metrics] running GREEN (ours, public prompt)...", file=sys.stderr)
        out["green_ours_public_prompt"] = run_green(episodes, client, args.judge_model)

    if args.reasoning_quality:
        print("[score_judged_metrics] running Table 2 reasoning quality (ours, published rubric)...",
              file=sys.stderr)
        out["reasoning_quality_ours"] = run_reasoning_quality(episodes, client, args.judge_model)

    text = json.dumps(out, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
