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

# Their reported numbers live in echosonar_r.py, re-exported here because every
# caller that needs the mapping also needs the table to compare against.
from echo_verl.eval.echosonar_r import (ECHOSONAR_R_MACRO12,  # noqa: E402,F401
                                        ECHOSONAR_R_TABLE1, their_macro)


def disease_of(question: str):
    """Question text -> disease key, or None if nothing matches."""
    q = (question or "").lower()
    for name, pat in _RULES:
        if re.search(pat, q):
            return name
    return None
