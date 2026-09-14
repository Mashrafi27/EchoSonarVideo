"""EchoSonar-R's reported numbers (arXiv 2606.28164), transcribed from the PDF.

Hard-coded rather than parsed on demand because arXiv is unreachable from this
cluster: the site proxy 403s the CONNECT to arxiv.org, export.arxiv.org and the
mirrors, both from the login node and through the agent harness. The PDF lives at
`2606.28164v1.pdf` in the repo root and is deliberately NOT committed.

Read the protocol notes before putting any of our numbers next to these:

TABLE 1 (abnormality classification) is PER-DISEASE positive-class F1 and
balanced accuracy, macro-averaged over 12 categories, on all 1,215 private test
studies. It is NOT a pooled yes/no score over a question mix. Our test file asks
11 of the 12 (no "Healthy" question), and our per-disease prevalences match
theirs to within 0.1%, so the same test set is confirmed -- but the macro must be
recomputed over the same 11 on both sides via `their_macro()`.

TABLE 2 (reasoning quality) scores reasoning traces 1-5 Likert across five
dimensions, judged by Mistral-7B on a prompt EchoSonar-R never published --
only the five dimension DEFINITIONS are (Sec. 3). `eval.
reasoning_quality` writes our own judge prompt from those definitions; treat
any comparison against ECHOSONAR_R_TABLE2 as directional, not a reproduction.

TABLE 3 (report generation) is on the private test set, full reports. BLEU/
METEOR/ROUGE-L are real implementations (`eval.nlg`, METEOR via
nltk) and BERTScore is a real implementation against PubMedBERT
(`eval.bertscore`) -- all three are faithfully comparable to these
numbers. GREEN is NOT: EchoSonar-R's echocardiography-adapted judge prompt was
never published, so `eval.green_approx` computes the *original*,
public GREEN prompt/formula instead -- a real metric, but not theirs. Label it
"GREEN (ours, public prompt)" and never place it unlabeled next to the 0.800
below.
"""

# disease -> {"prev": %, model: (F1, BAcc)}
ECHOSONAR_R_TABLE1 = {
    "tv_regurgitation":       {"prev": 54.6, "grpo": (66.6, 59.9), "sft": (66.6, 59.3), "qwen3vl": (38.3, 52.1)},
    "mv_regurgitation":       {"prev": 54.4, "grpo": (68.3, 61.6), "sft": (69.1, 62.6), "qwen3vl": (49.7, 51.5)},
    "la_enlargement":         {"prev": 27.3, "grpo": (64.9, 75.8), "sft": (62.7, 74.6), "qwen3vl": (26.6, 49.8)},
    "healthy":                {"prev": 17.5, "grpo": (38.0, 62.3), "sft": (36.2, 61.3), "qwen3vl": (20.1, 49.9)},
    "lv_systolic_dysfunction":{"prev": 16.3, "grpo": (59.5, 74.0), "sft": (55.9, 71.9), "qwen3vl": (21.8, 51.8)},
    "av_regurgitation":       {"prev": 15.4, "grpo": (42.3, 65.4), "sft": (41.3, 65.1), "qwen3vl": (24.2, 52.3)},
    "av_stenosis":            {"prev":  9.5, "grpo": (71.6, 82.2), "sft": (69.8, 80.9), "qwen3vl": (14.9, 50.3)},
    "mv_calcification":       {"prev":  8.6, "grpo": (45.4, 72.2), "sft": (44.5, 69.7), "qwen3vl": (15.4, 51.7)},
    "lv_enlargement":         {"prev":  5.7, "grpo": (57.4, 75.5), "sft": (53.9, 71.5), "qwen3vl": (11.2, 52.5)},
    "ra_enlargement":         {"prev":  4.8, "grpo": (25.9, 58.7), "sft": (18.4, 55.9), "qwen3vl": ( 6.0, 46.9)},
    "rv_enlargement":         {"prev":  2.5, "grpo": (13.8, 55.5), "sft": ( 3.6, 50.6), "qwen3vl": ( 2.5, 44.2)},
    "bicuspid_av":            {"prev":  2.5, "grpo": (39.2, 65.7), "sft": (19.2, 57.4), "qwen3vl": ( 4.5, 50.1)},
}
ECHOSONAR_R_MACRO12 = {"grpo": (49.4, 67.4), "sft": (45.1, 65.1), "qwen3vl": (19.6, 50.3)}

# Table 2: reasoning-trace quality, 1-5 Likert, judged by Mistral-7B on
# EchoSonar-R's (unpublished) prompt. dimension -> {model: score}.
ECHOSONAR_R_TABLE2 = {
    "reasoning_answer_agreement": {"grpo": 4.98, "sft": 4.96, "qwen3vl": 3.29, "medgemma": 3.33, "chiron_o1": 3.50, "lingshu": 4.02},
    "reasoning_efficiency":       {"grpo": 4.98, "sft": 4.97, "qwen3vl": 3.73, "medgemma": 3.74, "chiron_o1": 3.80, "lingshu": 4.58},
    "factual_correctness":        {"grpo": 4.99, "sft": 5.00, "qwen3vl": 4.77, "medgemma": 4.15, "chiron_o1": 3.49, "lingshu": 4.29},
    "evidence_grounding":         {"grpo": 5.00, "sft": 5.00, "qwen3vl": 3.31, "medgemma": 3.71, "chiron_o1": 3.56, "lingshu": 4.68},
    "terminology_accuracy":       {"grpo": 5.00, "sft": 5.00, "qwen3vl": 4.96, "medgemma": 4.29, "chiron_o1": 3.69, "lingshu": 4.70},
}
ECHOSONAR_R_TABLE2_AVERAGE = {"grpo": 4.99, "sft": 4.99, "qwen3vl": 4.01, "medgemma": 3.84, "chiron_o1": 3.61, "lingshu": 4.45}

# metric -> {model: value}. Their Qwen3-VL column is the untrained-base row our
# plain-prompt eval is meant to reproduce.
ECHOSONAR_R_TABLE3 = {
    "BLEU-1":    {"grpo": 0.795, "sft": 0.792, "qwen3vl": 0.076, "medgemma": 0.135, "echoprime": 0.143, "chiron_o1": 0.125, "lingshu": 0.099},
    "BLEU-2":    {"grpo": 0.767, "sft": 0.762, "qwen3vl": 0.027, "medgemma": 0.066, "echoprime": 0.061, "chiron_o1": 0.039, "lingshu": 0.031},
    "BLEU-3":    {"grpo": 0.740, "sft": 0.736, "qwen3vl": 0.014, "medgemma": 0.047, "echoprime": 0.036, "chiron_o1": 0.023, "lingshu": 0.016},
    "BLEU-4":    {"grpo": 0.725, "sft": 0.720, "qwen3vl": 0.010, "medgemma": 0.038, "echoprime": 0.027, "chiron_o1": 0.019, "lingshu": 0.011},
    "METEOR":    {"grpo": 0.829, "sft": 0.826, "qwen3vl": 0.195, "medgemma": 0.260, "echoprime": 0.234, "chiron_o1": 0.197, "lingshu": 0.230},
    "ROUGE-L":   {"grpo": 0.819, "sft": 0.815, "qwen3vl": 0.113, "medgemma": 0.188, "echoprime": 0.192, "chiron_o1": 0.185, "lingshu": 0.162},
    "BERTScore": {"grpo": 0.985, "sft": 0.985, "qwen3vl": 0.924, "medgemma": 0.929, "echoprime": 0.933, "chiron_o1": 0.931, "lingshu": 0.927},
    "GREEN":     {"grpo": 0.800, "sft": 0.796, "qwen3vl": 0.216, "medgemma": 0.453, "echoprime": 0.306, "chiron_o1": 0.358, "lingshu": 0.491},
}
# Metrics we cannot faithfully reproduce THEIR number for. METEOR and
# BERTScore are now real implementations (eval.nlg / .bertscore) and
# belong on the same axis as their column. GREEN is different in kind: our
# version (eval.green_approx) is the original public GREEN, not
# their unpublished echocardiography-adapted prompt -- always label it
# separately rather than placing it in this same column.
NOT_IMPLEMENTED = ("GREEN",)

MODEL_LABELS = {"grpo": "EchoSonar-R (GRPO)", "sft": "EchoSonar-R (SFT-only)",
                "qwen3vl": "Qwen3-VL (their row)", "medgemma": "MedGemma",
                "echoprime": "EchoPrime", "chiron_o1": "Chiron-o1", "lingshu": "Lingshu"}


def their_macro(model: str, diseases) -> tuple:
    """THEIR macro recomputed over exactly the diseases we measured.

    Comparing our macro-over-11 against their published macro-12 would credit or
    penalise us for a Healthy row we never ask about.
    """
    rows = [ECHOSONAR_R_TABLE1[d][model] for d in diseases if d in ECHOSONAR_R_TABLE1]
    if not rows:
        return (float("nan"), float("nan"))
    return (sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows))
