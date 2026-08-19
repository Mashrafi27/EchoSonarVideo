"""Score an episodes file into a metrics report.

    python -m echo_verl.eval.score_eval build/eval_step100.jsonl

Separate from generation on purpose: recomputing a metric must never require
re-running the model. Metrics come from echo_rl.eval (CardioBench/EchoSonar-R
definitions), NOT from echo_rl.reward -- the reward is what we optimise, these
are what we report.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from echo_rl.data.answers import finding_set, parse_yes_no          # noqa: E402
from echo_rl.eval.metrics import (balanced_accuracy, bootstrap_ci,  # noqa: E402
                                  macro_f1, majority_baseline, per_class_f1,
                                  set_f1, tool_call_rate, tools_per_episode,
                                  accuracy)
from echo_rl.eval.nlg import nlg_report                             # noqa: E402
from echo_rl.reward.sections import section_coverage                # noqa: E402

TEXT_TYPES = {"structure_description", "conclusion", "full_report"}


def score(episodes: list) -> dict:
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep["question_type"]].append(ep)

    out = {"n_episodes": len(episodes), "by_question_type": {}}

    # --- classification (yes/no) ---
    yn = by_type.get("abnormality_classification", [])
    if yn:
        y_true, y_pred = [], []
        for ep in yn:
            t = (ep["reward_key"] or {}).get("target")
            p = parse_yes_no(ep.get("answer") or "")
            if t is None:
                continue
            # An unparsable answer counts as WRONG, not as missing. Dropping it
            # would let a model that never answers score 100% on what remains.
            y_true.append(t)
            y_pred.append(p if p is not None else f"__unparsed_{t}__")
        res = {
            "n": len(y_true),
            "accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "per_class_f1": per_class_f1(y_true, y_pred),
            "majority_baseline": majority_baseline(y_true),
            "unparsable": sum(1 for p in y_pred if str(p).startswith("__unparsed_")),
        }
        res["balanced_accuracy_ci"] = bootstrap_ci(y_true, y_pred, balanced_accuracy)
        res["macro_f1_ci"] = bootstrap_ci(y_true, y_pred, macro_f1)
        out["by_question_type"]["abnormality_classification"] = res

    # --- multi-label findings ---
    al = by_type.get("abnormality_list", [])
    if al:
        scores = []
        for ep in al:
            gold = set((ep["reward_key"] or {}).get("target") or [])
            pred = finding_set(ep.get("answer") or "")
            scores.append(set_f1(pred, gold))
        out["by_question_type"]["abnormality_list"] = {
            "n": len(scores), "set_f1": sum(scores) / len(scores)}

    # --- free text ---
    for qt in TEXT_TYPES:
        eps = by_type.get(qt, [])
        if not eps:
            continue
        preds = [ep.get("answer") or "" for ep in eps]
        golds = [ep.get("gold_answer") or "" for ep in eps]
        res = nlg_report(preds, golds)
        if qt in ("full_report", "conclusion"):
            cov = [section_coverage(p, g) for p, g in zip(preds, golds)]
            res["section_coverage"] = sum(cov) / len(cov)
        res["empty_answers"] = sum(1 for p in preds if not p.strip())
        out["by_question_type"][qt] = res

    # --- agentic behaviour (across all types) ---
    traces = [ep.get("tool_calls") or [] for ep in episodes]
    ops = defaultdict(int)
    failed = 0
    for t in traces:
        for c in t:
            ops[c.get("op") or "<none>"] += 1
            if not c.get("ok"):
                failed += 1
    total_calls = sum(len(t) for t in traces)
    out["agentic"] = {
        "tool_call_rate": tool_call_rate(traces),
        "tools_per_episode": tools_per_episode(traces),
        "total_tool_calls": total_calls,
        "failed_tool_calls": failed,
        "failed_fraction": failed / total_calls if total_calls else 0.0,
        "ops": dict(ops),
        "malformed_tool_calls": sum(ep.get("malformed_tool_calls", 0) for ep in episodes),
        "finish_reasons": _counts(ep.get("finish_reason") for ep in episodes),
        "no_answer": sum(1 for ep in episodes if not ep.get("answer")),
    }
    return out


def _counts(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return dict(d)


def _flatten(report, prefix="eval"):
    """Nested report -> flat {dotted.key: number} for wandb scalars.

    Only numbers survive. Strings and lists (finish-reason breakdowns, per-class
    label names) are kept out of the scalar namespace deliberately: a chart of a
    string is noise, and they are already in the JSON artifact.
    """
    flat = {}
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}/{k}")
        elif isinstance(node, bool):
            flat[path] = int(node)
        elif isinstance(node, (int, float)) and node is not None:
            flat[path] = node
    walk(report, prefix)
    return flat


def log_to_wandb(report, episodes, *, project, name, run_config=None):
    """Log one evaluation as its own wandb run. Returns the run URL.

    Evaluation is a separate run rather than extra steps on the training run:
    an eval is repeated against many checkpoints and prompt settings, and folding
    those into the training timeline makes both unreadable. The checkpoint under
    test goes in the config so runs stay comparable.
    """
    import wandb

    run = wandb.init(project=project, name=name, job_type="eval",
                     config=run_config or {})
    run.summary.update(_flatten(report))
    # The episode table is what makes a bad number diagnosable -- without the
    # predictions and tool traces, a low score says nothing about why.
    columns = ["question_type", "finish_reason", "turns", "n_tool_calls",
               "images_used", "question", "gold_answer", "answer"]
    table = wandb.Table(columns=columns)
    for ep in episodes:
        table.add_data(
            ep.get("question_type"), ep.get("finish_reason"), ep.get("turns"),
            len(ep.get("tool_calls") or []), ep.get("images_used"),
            str(ep.get("question"))[:500], str(ep.get("gold_answer"))[:500],
            str(ep.get("answer"))[:500])
    run.log({"eval/episodes": table})
    url = run.url
    run.finish()
    return url


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--wandb", action="store_true", help="also log this eval to wandb")
    ap.add_argument("--wandb-project", default="echo-eval")
    ap.add_argument("--wandb-name", default=None,
                    help="run name (default: the episodes filename stem)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint under test; recorded in the wandb run config")
    args = ap.parse_args(argv)

    episodes = [json.loads(l) for l in open(args.episodes) if l.strip()]
    report = score(episodes)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    if args.wandb:
        url = log_to_wandb(
            report, episodes,
            project=args.wandb_project,
            name=args.wandb_name or Path(args.episodes).stem,
            run_config={"checkpoint": args.checkpoint,
                        "episodes_file": args.episodes,
                        "n_episodes": len(episodes)})
        print(f"logged to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
