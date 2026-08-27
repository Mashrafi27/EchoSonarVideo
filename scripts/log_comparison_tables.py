#!/usr/bin/env python3
"""Log every cross-run comparison we have as wandb TABLES, in one place.

    python scripts/log_comparison_tables.py \
        --run "base plain:build/eval_qwen3vl8b-base-plain_160066/report.json" \
        --run "SFT step616:build/eval_s5-step616_156202/report.json" \
        --wandb-name comparison-2026-08-27

Individual eval jobs each log their own run, which makes any comparison a matter
of opening several tabs and trusting your memory. This puts our runs and
EchoSonar-R's reported columns in single tables you can read down.

Three tables, plus whatever per-disease runs are passed:
  report_generation    our NLG numbers beside their Table 3
  classification       our POOLED yes/no numbers, clearly labelled as NOT
                       comparable to their Table 1 (which is a per-disease macro)
  agentic              tool behaviour, which has no counterpart in their work

Metrics we do not compute (METEOR, BERTScore, GREEN) appear as "not computed",
never as 0.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from echo_verl.eval.echosonar_r import (ECHOSONAR_R_TABLE3,    # noqa: E402
                                        MODEL_LABELS, NOT_IMPLEMENTED)


def _cell(v, nd=4):
    """Every cell in these tables is a STRING.

    wandb.Table infers a type per column and rejects a mixed column outright
    ("String not assignable to Number"), and these columns genuinely mix: a
    metric we do not compute has no number, and "not computed" must not be
    rendered as 0. These tables are for reading, not for plotting.
    """
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)

NLG_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L",
               "BERTScore", "GREEN"]
# Their Table 3 is full-report generation, so ours has to be the same task.
REPORT_TYPE = "full_report"


def _parse_run(spec):
    label, _, path = spec.partition(":")
    if not path:
        raise SystemExit(f"--run needs 'label:path/to/report.json', got {spec!r}")
    return label.strip(), json.loads(Path(path).read_text())


def build_tables(runs, per_disease):
    import wandb
    tables = {}

    cols = ["metric"] + [lbl for lbl, _ in runs] + \
           [MODEL_LABELS[m] for m in ("grpo", "sft", "qwen3vl")]
    t = wandb.Table(columns=cols)
    for metric in NLG_METRICS:
        row = [metric]
        for _, rep in runs:
            sec = rep.get("by_question_type", {}).get(REPORT_TYPE, {})
            v = sec.get(metric)
            row.append("not computed" if metric in NOT_IMPLEMENTED else _cell(v))
        row += [_cell(ECHOSONAR_R_TABLE3[metric][m]) for m in ("grpo", "sft", "qwen3vl")]
        t.add_data(*row)
    tables["comparison/report_generation"] = t

    cols = ["metric"] + [lbl for lbl, _ in runs]
    t = wandb.Table(columns=cols)
    yn_keys = [("n", "n"), ("accuracy", "accuracy"),
               ("balanced_accuracy", "balanced_accuracy"),
               ("macro_f1 (over yes/no)", "macro_f1"), ("unparsable", "unparsable")]
    for label, key in yn_keys:
        row = [label]
        for _, rep in runs:
            v = rep.get("by_question_type", {}).get("abnormality_classification", {}).get(key)
            row.append(_cell(v))
        t.add_data(*row)
    t.add_data("set_f1 (abnormality_list)",
               *[_cell(rep.get("by_question_type", {}).get("abnormality_list", {}).get("set_f1"))
                 for _, rep in runs])
    tables["comparison/classification_pooled"] = t

    cols = ["metric"] + [lbl for lbl, _ in runs]
    t = wandb.Table(columns=cols)
    for label, key in [("tool_call_rate", "tool_call_rate"),
                       ("tools_per_episode", "tools_per_episode"),
                       ("total_tool_calls", "total_tool_calls"),
                       ("failed_fraction", "failed_fraction"),
                       ("answered_with_tag", "answered_with_tag"),
                       ("no_output", "no_output")]:
        t.add_data(label, *[_cell(rep.get("agentic", {}).get(key)) for _, rep in runs])
    for op in ("select_view", "select_frames", "zoom"):
        t.add_data(f"op:{op}",
                   *[_cell(rep.get("agentic", {}).get("ops", {}).get(op, 0)) for _, rep in runs])
    tables["comparison/agentic"] = t

    for label, rep in per_disease:
        cols = ["disease", "prev_ours", "prev_theirs", "n", "positives",
                "predicted_yes", "our_F1", "our_BAcc",
                "R_GRPO_F1", "R_GRPO_BAcc", "R_SFT_F1", "R_SFT_BAcc"]
        t = wandb.Table(columns=cols)
        from echo_verl.eval.echosonar_r import ECHOSONAR_R_TABLE1
        by = rep["by_disease"]
        for d in sorted(by, key=lambda x: -by[x]["prevalence"]):
            r, their = by[d], ECHOSONAR_R_TABLE1.get(d, {})
            g, s = their.get("grpo", (None, None)), their.get("sft", (None, None))
            t.add_data(d, _cell(r["prevalence"], 1), _cell(their.get("prev"), 1),
                       _cell(r["n"]), _cell(r["positives"]), _cell(r["predicted_yes"]),
                       _cell(r["F1"], 1), _cell(r["BAcc"], 1),
                       _cell(g[0], 1), _cell(g[1], 1), _cell(s[0], 1), _cell(s[1], 1))
        tables[f"comparison/per_disease [{label}]"] = t
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="'label:path/to/report.json' from score_eval; repeatable")
    ap.add_argument("--per-disease", action="append", default=[],
                    help="'label:path/to/per_disease.json'; repeatable")
    ap.add_argument("--wandb-project", default="echo-eval")
    ap.add_argument("--wandb-name", default="comparison")
    args = ap.parse_args(argv)

    if not args.run:
        ap.error("give at least one --run")
    runs = [_parse_run(s) for s in args.run]
    per_disease = [_parse_run(s) for s in args.per_disease]

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                     job_type="comparison",
                     config={"runs": [l for l, _ in runs],
                             "per_disease_runs": [l for l, _ in per_disease],
                             "reference": "EchoSonar-R arXiv 2606.28164 Tables 1 and 3"})
    run.log(build_tables(runs, per_disease))
    print(f"logged to {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
