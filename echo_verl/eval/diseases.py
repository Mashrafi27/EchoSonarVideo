"""Map a binary abnormality question to the disease it asks about.

EchoSonar-R (arXiv 2606.28164) does NOT report a pooled yes/no score. Their
Table 1 is per-disease positive-class F1 and balanced accuracy, macro-averaged
over abnormality categories, on all 1,215 private test studies. Our pooled
metric is a different quantity and cannot be placed on their scale, so a
comparable row needs the disease behind each question.

The test set asks each disease through several phrasings (44 phrasings, 11
diseases). Matching is by keyword on the question text, ORDER MATTERS: a
systolic-dysfunction question also contains "left ventric", so it has to be
tested before the LV-enlargement rule.

Their 12th category, "Healthy", has no question of its own in our test file.
Compare macro-over-11 to a macro-over-11 recomputed from their per-disease
cells, never to their published macro-12.
"""
import re

# (disease, pattern). First match wins.
_RULES = [
    ("bicuspid_av",             r"bicuspid"),
    ("lv_systolic_dysfunction", r"systolic"),
    ("mv_calcification",        r"calcification"),
    ("av_stenosis",             r"aortic (valve )?stenosis|stenosis"),
    ("av_regurgitation",        r"aortic (valve )?regurgitation"),
    ("mv_regurgitation",        r"mitral (valve )?regurgitation"),
    ("tv_regurgitation",        r"tricuspid"),
    ("la_enlargement",          r"left atri"),
    ("ra_enlargement",          r"right atri"),
    ("lv_enlargement",          r"left ventric"),
    ("rv_enlargement",          r"right ventric"),
]

# Their Table 1, private test set, per-disease cells. Kept here so the
# comparison never again blocks on fetching a proxy-blocked PDF.
# disease -> {model: (F1, BAcc)}; prevalence in percent.
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
# Their published macro is over all 12 rows above.
ECHOSONAR_R_MACRO12 = {"grpo": (49.4, 67.4), "sft": (45.1, 65.1), "qwen3vl": (19.6, 50.3)}


def disease_of(question: str):
    """Question text -> disease key, or None if nothing matches."""
    q = (question or "").lower()
    for name, pat in _RULES:
        if re.search(pat, q):
            return name
    return None


def their_macro(model: str, diseases) -> tuple:
    """Recompute THEIR macro over exactly the diseases we measured.

    Comparing our macro-over-11 against their published macro-12 would credit or
    penalise us for a Healthy row we never asked about.
    """
    rows = [ECHOSONAR_R_TABLE1[d][model] for d in diseases if d in ECHOSONAR_R_TABLE1]
    if not rows:
        return (float("nan"), float("nan"))
    return (sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows))
