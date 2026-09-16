"""Closed 11-canonical-finding taxonomy for `abnormality_list` outcome scoring, ported
verbatim from report_generation/grpo/rewards.py (Darya's own GRPO reward) rather than this
project's previous open free-text bullet extraction (data_core.data.answers.finding_set) --
a closed vocabulary + alias table is meaningfully harder to game than raw substring presence
(spec: docs/superpowers/specs/2026-09-15-echoprime-tool-grpo-design.md section 5, item 2).

Also the same 11 categories packages/eval/diseases.py::disease_of already uses at question
level (its own rule table is QUESTION-text keyword rules; this one matches FINDING-text
inside a list answer -- different vocabulary distribution, kept as a separate table
deliberately, not merged).
"""
import re

CANONICAL_FINDINGS = {
    "tricuspid regurgitation",
    "mitral valve regurgitation",
    "left atrial enlargement",
    "aortic regurgitation",
    "left ventricular systolic function",
    "mitral valve calcification",
    "left ventricular enlargement",
    "aortic stenosis",
    "right atrial enlargement",
    "bicuspid aortic valve",
    "right ventricular enlargement",
}

_ALIASES = {
    "tricuspid regurgitation": ["tricuspid regurgitation", "tr"],
    "mitral valve regurgitation": ["mitral valve regurgitation", "mitral regurgitation", "mr"],
    "left atrial enlargement": ["left atrial enlargement", "la enlargement", "left atrium enlargement"],
    "aortic regurgitation": ["aortic regurgitation", "ar", "aortic insufficiency"],
    "left ventricular systolic function": [
        "left ventricular systolic function", "lv systolic function",
        "left ventricular systolic dysfunction", "lv systolic dysfunction",
    ],
    "mitral valve calcification": ["mitral valve calcification", "mitral calcification"],
    "left ventricular enlargement": ["left ventricular enlargement", "lv enlargement"],
    "aortic stenosis": ["aortic stenosis", "as"],
    "right atrial enlargement": ["right atrial enlargement", "ra enlargement", "ra dilation", "right atrial dilation"],
    "bicuspid aortic valve": ["bicuspid aortic valve", "bav"],
    "right ventricular enlargement": [
        "right ventricular enlargement", "rv enlargement", "rv dilation", "right ventricular dilation",
    ],
}

_ALIAS_TO_CANONICAL = {}
for _canonical, _aliases in _ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical

_NO_ABNORMALITY_RE = re.compile(r"no\s+(significant\s+)?abnormalit", re.I)

# RT-DETR classes, matching report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES
# exactly (0=Left Ventricle, 1=Left Atrium, 2=Right Atrium, 3=Right Ventricle,
# 4=Mitral Valve, 5=Tricuspid Valve, 6=LVOT Area). None = no RT-DETR class covers this
# finding's structure at all (e.g. every aortic-valve finding -- RT-DETR has no aortic valve
# class) -- these contribute nothing to the grounding average (Task 4), never penalized.
FINDING_TO_DETR_CLASS = {
    "left ventricular enlargement": 0,
    "left ventricular systolic function": 0,
    "left atrial enlargement": 1,
    "right atrial enlargement": 2,
    "right ventricular enlargement": 3,
    "mitral valve regurgitation": 4,
    "mitral valve calcification": 4,
    "tricuspid regurgitation": 5,
    "aortic regurgitation": None,
    "aortic stenosis": None,
    "bicuspid aortic valve": None,
}


def extract_canonical_findings(text: str) -> set:
    """Free-text list answer -> set of canonical finding names. Bullets that don't match any
    known alias are DROPPED (not kept as open free text) -- the closed vocabulary is the
    entire point (see module docstring)."""
    text_lower = (text or "").lower()
    if _NO_ABNORMALITY_RE.search(text_lower):
        return set()

    text_lower = re.sub(r"the following abnormalities are identified:?\s*", "", text_lower)
    text_lower = re.sub(r"\d+\.\s*", "", text_lower)
    items = re.split(r"[-•\n;,]+", text_lower)

    found = set()
    for item in items:
        cleaned = item.strip().rstrip(".")
        if not cleaned or len(cleaned) < 2:
            continue
        if cleaned in _ALIAS_TO_CANONICAL:
            found.add(_ALIAS_TO_CANONICAL[cleaned])
            continue
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
            if len(alias) <= 3:
                continue
            if alias in cleaned or cleaned in alias:
                found.add(canonical)
                break
    return found


def iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
