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

BLEU/METEOR/ROUGE-L come from --run report.json files; BERTScore and GREEN
(ours, public prompt -- NOT EchoSonar-R's unpublished one) come from separate
--judged report.json files (score_judged_metrics.py); --reasoning-quality adds
a Table 2 comparison (also ours, from their published rubric text). Anything
genuinely uncomputed appears as "not computed", never as 0.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages"))

from eval.echosonar_r import (ECHOSONAR_R_TABLE2, ECHOSONAR_R_TABLE2_AVERAGE,  # noqa: E402
                            ECHOSONAR_R_TABLE3, MODEL_LABELS, NOT_IMPLEMENTED)


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

# From score_eval.py's report.json (by_question_type.full_report.<metric>):
# real, faithfully-comparable numbers, directly on EchoSonar-R's Table 3 axis.
NLG_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"]
# From score_judged_metrics.py's --json-out (a separate file, since these need
# a model/judge score_eval.py doesn't load): BERTScore is comparable the same
# way; GREEN is NOT (see its own table below) and is deliberately absent here.
JUDGED_NLG_METRICS = [("BERTScore", "bertscore", "f1")]
# Their Table 3 is full-report generation, so ours has to be the same task.
REPORT_TYPE = "full_report"


def _parse_run(spec):
    label, _, path = spec.partition(":")
    if not path:
        raise SystemExit(f"--run needs 'label:path/to/report.json', got {spec!r}")
    return label.strip(), json.loads(Path(path).read_text())


def build_tables(runs, per_disease, judged=None, reasoning=None):
    import wandb
    tables = {}
    judged = dict(judged or [])

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
    for label_name, judged_key, sub_key in JUDGED_NLG_METRICS:
        row = [label_name]
        for lbl, _ in runs:
            v = judged.get(lbl, {}).get(judged_key, {}).get(sub_key)
            row.append(_cell(v) if v is not None else "not computed")
        row += [_cell(ECHOSONAR_R_TABLE3[label_name][m]) for m in ("grpo", "sft", "qwen3vl")]
        t.add_data(*row)
    row = ["GREEN"] + ["not computed (theirs unreproducible -- see green_ours table)"
                        for _ in runs]
    row += [_cell(ECHOSONAR_R_TABLE3["GREEN"][m]) for m in ("grpo", "sft", "qwen3vl")]
    t.add_data(*row)
    tables["comparison/report_generation"] = t

    if judged:
        cols = ["run", "GREEN (ours, public prompt)", "n"]
        t = wandb.Table(columns=cols)
        for lbl, _ in runs:
            g = judged.get(lbl, {}).get("green_ours_public_prompt")
            if g is not None:
                t.add_data(lbl, _cell(g.get("green_ours_public_prompt")), g.get("n"))
        tables["comparison/green_ours_public_prompt"] = t

    if reasoning:
        dims = list(ECHOSONAR_R_TABLE2.keys())
        cols = ["dimension"] + [f"{lbl} (ours)" for lbl, _ in reasoning] + \
               [f"{MODEL_LABELS[m]} (their prompt)" for m in
                ("grpo", "sft", "qwen3vl", "medgemma", "chiron_o1", "lingshu")]
        t = wandb.Table(columns=cols)
        for dim in dims:
            row = [dim]
            for _, rep in reasoning:
                row.append(_cell(rep.get("means", {}).get(dim)))
            row += [_cell(ECHOSONAR_R_TABLE2[dim][m]) for m in
                    ("grpo", "sft", "qwen3vl", "medgemma", "chiron_o1", "lingshu")]
            t.add_data(*row)
        row = ["average"]
        for _, rep in reasoning:
            row.append(_cell(rep.get("means", {}).get("average")))
        row += [_cell(ECHOSONAR_R_TABLE2_AVERAGE[m]) for m in
                ("grpo", "sft", "qwen3vl", "medgemma", "chiron_o1", "lingshu")]
        t.add_data(*row)
        tables["comparison/reasoning_quality_ours_vs_table2"] = t

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
        from eval.echosonar_r import ECHOSONAR_R_TABLE1
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
    ap.add_argument("--judged", action="append", default=[],
                    help="'label:path/to/judged.json' from score_judged_metrics.py "
                         "(BERTScore + GREEN-ours); label MUST match a --run label")
    ap.add_argument("--reasoning-quality", action="append", default=[],
                    help="'label:path/to/reasoning_quality.json' (the "
                         "'reasoning_quality_ours' key from score_judged_metrics.py, "
                         "or that file directly); repeatable, Table 2 comparison")
    ap.add_argument("--wandb-project", default="echo-eval")
    ap.add_argument("--wandb-name", default="comparison")
    args = ap.parse_args(argv)

    if not args.run:
        ap.error("give at least one --run")
    runs = [_parse_run(s) for s in args.run]
    per_disease = [_parse_run(s) for s in args.per_disease]
    judged = [_parse_run(s) for s in args.judged]
    reasoning = []
    for s in args.reasoning_quality:
        label, rep = _parse_run(s)
        # accept either score_judged_metrics.py's full --json-out (which nests
        # this under "reasoning_quality_ours") or a file that IS that object
        reasoning.append((label, rep.get("reasoning_quality_ours", rep)))

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                     job_type="comparison",
                     config={"runs": [l for l, _ in runs],
                             "per_disease_runs": [l for l, _ in per_disease],
                             "judged_runs": [l for l, _ in judged],
                             "reasoning_quality_runs": [l for l, _ in reasoning],
                             "reference": "EchoSonar-R arXiv 2606.28164 Tables 1-3"})
    run.log(build_tables(runs, per_disease, judged=judged, reasoning=reasoning))
    print(f"logged to {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
