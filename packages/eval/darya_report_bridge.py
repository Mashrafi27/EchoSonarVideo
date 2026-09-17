"""Wire our full_report episodes into Darya's own report_generation eval pipeline
(`report_generation/evaluation/report_pipeline/evaluate_reports.py`), instead of
scoring report generation with our own approximation in `eval.nlg`.

Why this exists: `eval.nlg` scores WHOLE reports, corpus-level. Her pipeline scores
per-SECTION (13 named cardiac structures + conclusions), sentence-level, and that is
what EchoSonar-R's Table 3 numbers were actually computed with -- confirmed by reading
her evaluate_reports.py directly, not inferred. Whole-report BLEU and her per-section
BLEU are not the same number even on identical text, so our earlier `eval.nlg` numbers
next to Table 3 were comparing apples to a different fruit that happens to also be
red. This module makes the comparison real.

Two steps, mirroring her own `convert_predictions.py` + `run_reports.py`:
  1. Parse our model's free-text `full_report` answers into her section dict, with
     her EXACT parser (`report_prompts.parse_report_sections`, the same "**Header:**"
     regex her own run_reports.py applies to her own models' output). We do not
     write a second parser: a report that doesn't match her format is legitimately
     un-scorable at the section level, not a bug in the bridge. On the 187793 base
     Qwen3-VL run, 808/1184 reports parsed (the rest are free prose with no bold
     headers) -- report that count every time, never silently drop it.
  2. Call her `evaluate_reports.evaluate_model()` UNMODIFIED against her own
     `test.json` ground truth (same 1,215-study test set: study_uuid overlap
     confirmed 1184/1184 on the full_report split) by pointing her module-level
     PREDICTIONS_DIR/RESULTS_DIR at ours. Her scoring code (ROUGE/BLEU/METEOR/
     BERTScore, all per-section) never gets copied or reimplemented here.

Usage:
    python -m eval.darya_report_bridge build/eval_x/episodes.jsonl \\
        --model-name qwen3vl_base_plain --json-out build/eval_x/darya_report_metrics.json

Requires report_generation/ on this machine (DARYA_REPORT_PIPELINE_DIR env var to
override its location; defaults to the known path on this cluster).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.score_eval import resolve_answer  # noqa: E402

DEFAULT_DARYA_DIR = "/vast/users/mohammad.yaqub/report_generation/evaluation/report_pipeline"
# Her evaluate_reports.py hardcodes GT_PATH to her own account's path
# (/data/daryataratynova/reports_v4/test.json), which doesn't exist on this cluster.
# Same test.json content lives here instead (1,215 studies, study_uuid overlap with
# our full_report episodes confirmed 1184/1184).
DEFAULT_GT_PATH = "/vast/users/mohammad.yaqub/report_generation/data/test.json"


def _load_darya_modules(darya_dir: str):
    """Import her report_prompts.py and evaluate_reports.py by path, without adding
    her directory to sys.path permanently (avoids shadowing our own `eval` package,
    since her repo also has generically-named modules)."""
    def _load(name, filename):
        spec = importlib.util.spec_from_file_location(name, os.path.join(darya_dir, filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # evaluate_reports doesn't import report_prompts, so no ordering issue
        spec.loader.exec_module(mod)
        return mod

    report_prompts = _load("_darya_report_prompts", "report_prompts.py")
    evaluate_reports = _load("_darya_evaluate_reports", "evaluate_reports.py")
    return report_prompts, evaluate_reports


def build_predictions_file(episodes_path: str, model_name: str, out_dir: str,
                            report_prompts) -> dict:
    """Parse our full_report episodes into her {study_id, sections} schema and write
    `{model_name}_reports.jsonl` into out_dir (her PREDICTIONS_DIR convention).

    Returns a stats dict: {n_total, n_parsed, n_unparsed}.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_reports.jsonl")
    n_total = n_parsed = 0
    with open(episodes_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep.get("question_type") != "full_report":
                continue
            n_total += 1
            text = resolve_answer(ep)
            sections = report_prompts.parse_report_sections(text)
            if sections:
                n_parsed += 1
            fout.write(json.dumps({"study_id": ep["study_uuid"], "sections": sections}) + "\n")
    return {"n_total": n_total, "n_parsed": n_parsed, "n_unparsed": n_total - n_parsed,
            "predictions_path": out_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", help="episodes.jsonl from run_eval.py (needs full_report rows)")
    ap.add_argument("--model-name", required=True,
                    help="label for this run; becomes <model-name>_reports.jsonl")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip BERTScore (matches her --no-bertscore, much faster)")
    ap.add_argument("--darya-dir", default=os.environ.get("DARYA_REPORT_PIPELINE_DIR",
                                                            DEFAULT_DARYA_DIR))
    ap.add_argument("--gt-path", default=os.environ.get("DARYA_REPORT_GT_PATH",
                                                          DEFAULT_GT_PATH))
    ap.add_argument("--work-dir", default=None,
                    help="where to write <model-name>_reports.jsonl (default: alongside episodes)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.darya_dir):
        raise SystemExit(f"Darya's report_pipeline not found at {args.darya_dir} -- "
                          f"set DARYA_REPORT_PIPELINE_DIR or --darya-dir")

    report_prompts, evaluate_reports = _load_darya_modules(args.darya_dir)

    work_dir = args.work_dir or os.path.dirname(os.path.abspath(args.episodes))
    stats = build_predictions_file(args.episodes, args.model_name, work_dir, report_prompts)
    print(f"[darya_report_bridge] parsed {stats['n_parsed']}/{stats['n_total']} reports into "
          f"her section format ({stats['n_unparsed']} had no \"**Header:**\" markers and are "
          f"NOT scored at the section level -- this is expected for plain-prompt output, not "
          f"an error).")

    # Point her module-level globals at our paths. GT_PATH stays hers: same test set
    # (study_uuid overlap confirmed 1184/1184 on this split), no reason to duplicate it.
    evaluate_reports.PREDICTIONS_DIR = work_dir
    evaluate_reports.RESULTS_DIR = work_dir
    evaluate_reports.GT_PATH = args.gt_path

    gt = evaluate_reports.load_ground_truth()
    print(f"[darya_report_bridge] loaded {len(gt)} studies with report sections from her test.json")

    results = evaluate_reports.evaluate_model(args.model_name, gt,
                                              use_bertscore=not args.no_bertscore)
    if results is None:
        raise SystemExit("evaluate_model returned no results -- check n_matched above")

    results["_bridge_stats"] = stats
    evaluate_reports.print_results(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"[darya_report_bridge] wrote {args.json_out}")


if __name__ == "__main__":
    main()
