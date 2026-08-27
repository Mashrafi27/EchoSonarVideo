"""Per-disease classification metrics, in EchoSonar-R Table 1's shape.

    python -m echo_verl.eval.score_per_disease build/eval_.../episodes.jsonl --model sft

Why this exists alongside score_eval.py: `score_eval` reports ONE pooled yes/no
score over a frequency-weighted mix of questions. EchoSonar-R reports per-disease
positive-class F1 and balanced accuracy, macro-averaged over abnormality
categories. Those are different quantities and one cannot be read off the other,
so a pooled 0.63 BAcc says nothing about their 50.3.

Two rules this file exists to enforce:
  - F1 is the POSITIVE class only, per disease. Not macro over {yes, no}.
  - Our macro is over the 11 diseases our test file asks about; their published
    macro is over 12 and includes a Healthy row we never ask. The comparison
    column recomputes THEIR macro over the same 11.

Unparsable predictions count as WRONG rather than being dropped. A model that
answers unintelligibly has not classified anything, and dropping those rows
would flatter a model that fails to follow the format.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from echo_rl.data.answers import parse_yes_no                       # noqa: E402
from echo_rl.eval.metrics import balanced_accuracy, bootstrap_ci    # noqa: E402
from echo_verl.eval.diseases import (ECHOSONAR_R_TABLE1,            # noqa: E402
                                     disease_of, their_macro)
from echo_verl.eval.score_eval import resolve_answer                # noqa: E402


def _f1_positive(gold, pred):
    """F1 of the 'yes' class. Undefined (0.0) when nothing is predicted or present."""
    tp = sum(1 for g, p in zip(gold, pred) if g == "yes" and p == "yes")
    fp = sum(1 for g, p in zip(gold, pred) if g != "yes" and p == "yes")
    fn = sum(1 for g, p in zip(gold, pred) if g == "yes" and p != "yes")
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _bacc(gold, pred):
    """Balanced accuracy with 'yes' as positive; unparsable predictions are wrong."""
    return balanced_accuracy([g == "yes" for g in gold],
                             [p == "yes" for p in pred])


def score(episodes: list) -> dict:
    rows = defaultdict(lambda: {"gold": [], "pred": []})
    skipped = 0
    for ep in episodes:
        if ep.get("question_type") != "abnormality_classification":
            continue
        d = disease_of(ep.get("question"))
        if d is None:
            skipped += 1
            continue
        rows[d]["gold"].append(parse_yes_no(str(ep.get("gold_answer"))))
        # parse_yes_no returns None on unintelligible output; kept as a non-'yes'
        # value so it scores as wrong instead of vanishing from the denominator.
        rows[d]["pred"].append(parse_yes_no(resolve_answer(ep)))

    out = {"by_disease": {}, "unmapped_questions": skipped}
    for d, r in rows.items():
        gold, pred = r["gold"], r["pred"]
        # CIs matter more here than anywhere else in this repo: at 2.5% prevalence
        # a 400-study sample holds ~10 positives, and an F1 built on 10 positives
        # looks like a number while carrying almost no information.
        f1_ci = bootstrap_ci(gold, pred, lambda g, p: 100 * _f1_positive(g, p))
        ba_ci = bootstrap_ci(gold, pred, lambda g, p: 100 * _bacc(g, p))
        out["by_disease"][d] = {
            "n": len(gold),
            "positives": sum(1 for g in gold if g == "yes"),
            "prevalence": 100 * sum(1 for g in gold if g == "yes") / len(gold),
            "predicted_yes": sum(1 for p in pred if p == "yes"),
            "unparsable": sum(1 for p in pred if p is None),
            "F1": 100 * _f1_positive(gold, pred),
            "BAcc": 100 * _bacc(gold, pred),
            "F1_ci": [f1_ci["ci_lo"], f1_ci["ci_hi"]],
            "BAcc_ci": [ba_ci["ci_lo"], ba_ci["ci_hi"]],
            "echosonar_r_prevalence": ECHOSONAR_R_TABLE1.get(d, {}).get("prev"),
        }

    ds = sorted(out["by_disease"])
    if ds:
        out["macro"] = {
            "n_diseases": len(ds),
            "F1": sum(out["by_disease"][d]["F1"] for d in ds) / len(ds),
            "BAcc": sum(out["by_disease"][d]["BAcc"] for d in ds) / len(ds),
        }
        # Their macro recomputed over OUR disease set, so the two columns are
        # averages of the same rows.
        out["echosonar_r_macro_same_diseases"] = {
            m: {"F1": round(f, 1), "BAcc": round(b, 1)}
            for m, (f, b) in ((m, their_macro(m, ds)) for m in ("grpo", "sft", "qwen3vl"))
        }
    return out


def render(report: dict) -> str:
    lines = []
    head = (f"{'disease':26s} {'prev%':>6s} {'n':>5s} {'pos':>5s} "
            f"{'F1':>6s} {'BAcc':>6s} | {'theirF1':>8s} {'theirBAcc':>10s}  (GRPO)")
    lines.append(head)
    lines.append("-" * len(head))
    for d in sorted(report["by_disease"], key=lambda x: -report["by_disease"][x]["prevalence"]):
        r = report["by_disease"][d]
        t = ECHOSONAR_R_TABLE1.get(d, {}).get("grpo", (float("nan"),) * 2)
        lines.append(f"{d:26s} {r['prevalence']:6.1f} {r['n']:5d} {r['positives']:5d} "
                     f"{r['F1']:6.1f} {r['BAcc']:6.1f} | {t[0]:8.1f} {t[1]:10.1f}")
    m = report.get("macro", {})
    lines.append("-" * len(head))
    lines.append(f"{'MACRO (ours)':26s} {'':6s} {'':5s} {'':5s} "
                 f"{m.get('F1', 0):6.1f} {m.get('BAcc', 0):6.1f}")
    for name, key in (("EchoSonar-R GRPO", "grpo"), ("EchoSonar-R SFT-only", "sft"),
                      ("Qwen3-VL (their row)", "qwen3vl")):
        t = report.get("echosonar_r_macro_same_diseases", {}).get(key, {})
        lines.append(f"{name:26s} {'':6s} {'':5s} {'':5s} "
                     f"{t.get('F1', 0):6.1f} {t.get('BAcc', 0):6.1f}")
    lines.append("")
    lines.append("Macros are over the same 11 diseases on both sides; their published "
                 "macro of 12 includes a Healthy row our test file never asks.")
    return "\n".join(lines)


def _fmt_ci(lo, hi):
    return "n/a" if lo is None else f"[{lo:.1f}, {hi:.1f}]"


def log_to_wandb(report, episodes, *, project, name, run_config=None):
    """Log the comparison as wandb TABLES, not just scalars.

    Scalars alone are unverifiable: a macro F1 of 31.2 gives no way to see which
    disease dragged it, whether the model simply never said yes, or whether the
    row rests on nine positives. Three tables, in the order you would check them:
    the Table 1 comparison, the macro summary, then every classification episode
    with its parsed prediction so a suspicious cell can be traced to answers.
    """
    import wandb

    run = wandb.init(project=project, name=name, job_type="eval-per-disease",
                     config=run_config or {})

    ds = sorted(report["by_disease"],
                key=lambda x: -report["by_disease"][x]["prevalence"])

    cmp_cols = ["disease", "prev_ours", "prev_theirs", "n", "positives",
                "predicted_yes", "unparsable",
                "our_F1", "our_F1_ci", "our_BAcc", "our_BAcc_ci",
                "R_GRPO_F1", "R_GRPO_BAcc", "R_SFT_F1", "R_SFT_BAcc",
                "Qwen3VL_F1", "Qwen3VL_BAcc",
                "dF1_vs_R_SFT", "dBAcc_vs_R_SFT"]
    cmp_table = wandb.Table(columns=cmp_cols)
    for d in ds:
        r = report["by_disease"][d]
        t = ECHOSONAR_R_TABLE1.get(d, {})
        grpo, sft, qwen = t.get("grpo", (None, None)), t.get("sft", (None, None)), t.get("qwen3vl", (None, None))
        cmp_table.add_data(
            d, round(r["prevalence"], 1), t.get("prev"), r["n"], r["positives"],
            r["predicted_yes"], r["unparsable"],
            round(r["F1"], 1), _fmt_ci(*r["F1_ci"]),
            round(r["BAcc"], 1), _fmt_ci(*r["BAcc_ci"]),
            grpo[0], grpo[1], sft[0], sft[1], qwen[0], qwen[1],
            None if sft[0] is None else round(r["F1"] - sft[0], 1),
            None if sft[1] is None else round(r["BAcc"] - sft[1], 1))
    run.log({"eval/per_disease": cmp_table})

    macro = report.get("macro", {})
    theirs = report.get("echosonar_r_macro_same_diseases", {})
    macro_table = wandb.Table(columns=["model", "macro_F1", "macro_BAcc", "n_diseases"])
    macro_table.add_data("ours (this run)", round(macro.get("F1", 0), 1),
                         round(macro.get("BAcc", 0), 1), macro.get("n_diseases"))
    for label, key in (("EchoSonar-R GRPO", "grpo"),
                       ("EchoSonar-R SFT-only", "sft"),
                       ("Qwen3-VL (their row)", "qwen3vl")):
        t = theirs.get(key, {})
        macro_table.add_data(label, t.get("F1"), t.get("BAcc"), macro.get("n_diseases"))
    run.log({"eval/macro_vs_echosonar_r": macro_table})

    ep_table = wandb.Table(columns=["disease", "study_uuid", "question", "gold",
                                    "predicted", "correct", "scored_text"])
    for ep in episodes:
        if ep.get("question_type") != "abnormality_classification":
            continue
        d = disease_of(ep.get("question"))
        if d is None:
            continue
        gold = parse_yes_no(str(ep.get("gold_answer")))
        text = resolve_answer(ep)
        pred = parse_yes_no(text)
        ep_table.add_data(d, ep.get("study_uuid"), str(ep.get("question"))[:300],
                          gold, pred, int(pred == gold), str(text)[:500])
    run.log({"eval/classification_episodes": ep_table})

    flat = {"eval/per_disease_macro/F1": macro.get("F1"),
            "eval/per_disease_macro/BAcc": macro.get("BAcc")}
    for d in ds:
        r = report["by_disease"][d]
        flat[f"eval/per_disease/{d}/F1"] = r["F1"]
        flat[f"eval/per_disease/{d}/BAcc"] = r["BAcc"]
        flat[f"eval/per_disease/{d}/predicted_yes"] = r["predicted_yes"]
    for label, key in (("grpo", "grpo"), ("sft", "sft"), ("qwen3vl", "qwen3vl")):
        t = theirs.get(key, {})
        flat[f"eval/echosonar_r/{label}/F1"] = t.get("F1")
        flat[f"eval/echosonar_r/{label}/BAcc"] = t.get("BAcc")
    run.summary.update({k: v for k, v in flat.items() if v is not None})

    url = run.url
    run.finish()
    return url


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes", nargs="+", help="one or more episodes.jsonl files")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--wandb", action="store_true", help="log the tables to wandb")
    ap.add_argument("--wandb-project", default="echo-eval")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)

    episodes = []
    for path in args.episodes:
        episodes += [json.loads(l) for l in open(path) if l.strip()]
    report = score(episodes)
    print(render(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    if args.wandb:
        url = log_to_wandb(
            report, episodes,
            project=args.wandb_project,
            name=args.wandb_name or Path(args.episodes[0]).parent.name,
            run_config={"checkpoint": args.checkpoint,
                        "episodes_files": args.episodes,
                        "n_episodes": len(episodes),
                        "protocol": "EchoSonar-R Table 1, per-disease macro"})
        print(f"logged to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
