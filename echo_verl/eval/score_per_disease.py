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
        out["by_disease"][d] = {
            "n": len(gold),
            "positives": sum(1 for g in gold if g == "yes"),
            "prevalence": 100 * sum(1 for g in gold if g == "yes") / len(gold),
            "predicted_yes": sum(1 for p in pred if p == "yes"),
            "unparsable": sum(1 for p in pred if p is None),
            "F1": 100 * _f1_positive(gold, pred),
            "BAcc": 100 * _bacc(gold, pred),
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes", nargs="+", help="one or more episodes.jsonl files")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    episodes = []
    for path in args.episodes:
        episodes += [json.loads(l) for l in open(path) if l.strip()]
    report = score(episodes)
    print(render(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
