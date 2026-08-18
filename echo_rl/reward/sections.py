"""Split a structured echo answer into its labelled sections.

Why this exists: `full_report` answers carry a median of 13 labelled sections
(Left Ventricle, Right Ventricle, ... Conclusions) and `conclusion` answers a
median of 4 sentences. Scoring either as ONE blob of text means getting the left
ventricle right and the right ventricle wrong is indistinguishable from the
reverse, and there is no gradient toward fixing any individual section.

EchoSonar-R (arXiv 2606.28164) scores reports as a mean over sections:
    r_cor^report = (1/|C|) * sum_c cos(e_pred_c, e_gt_c)
We keep that per-section shape, but score each section with whatever scorer suits
it rather than committing to an embedding model inside the reward path.

Measured on build/eval.jsonl (400 sampled full_report answers): 13 headers appear
in every single one, and non-header lines are always WRAPPED CONTINUATIONS of the
preceding section -- so they are appended, never dropped.
"""
import re

# "Left Ventricle: Normal dimensions..." — a capitalised label then a colon.
# Bounded length so a sentence containing a colon mid-prose is not mistaken for
# a header.
_HEADER = re.compile(r"^([A-Z][A-Za-z /()'\-]{2,40}):\s*(.*)$")


def split_sections(text: str) -> dict:
    """Labelled sections -> {normalised label: body}.

    Returns {} when the text has no headers at all, which is the caller's signal
    to fall back to whole-text scoring rather than silently score nothing.
    """
    sections, current = {}, None
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _HEADER.match(line)
        if m:
            current = _norm_label(m.group(1))
            body = m.group(2).strip()
            # A repeated header appends rather than overwriting.
            sections[current] = (sections.get(current, "") + " " + body).strip() if current in sections else body
        elif current is not None:
            sections[current] = (sections[current] + " " + line).strip()
    return sections


def _norm_label(label: str) -> str:
    return " ".join(label.split()).lower()


def split_sentences(text: str) -> list:
    """Sentence split for answers with no headers (e.g. `conclusion`).

    Deliberately simple: split on a period followed by whitespace. Clinical text
    here is short declarative sentences; a heavier tokenizer would add a
    dependency to the reward path for no measured gain.
    """
    parts = re.split(r"(?<=\.)\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def score_by_section(pred: str, gold: str, scorer, *, min_sections: int = 2):
    """Mean of per-section scores, aligned by section label.

    Recall-oriented on purpose: the denominator is the GOLD sections, so omitting
    a section scores 0 for it rather than being skipped. A model that emits only
    the one section it is confident about must not score 1.0.

    Extra sections the gold does not have are ignored -- padding the report with
    invented headers neither helps nor hurts here; inventing *content* is caught
    by the per-section scorer.

    Returns None when the gold has fewer than `min_sections` labelled sections,
    meaning this answer is not structured and the caller should score it whole.
    """
    gold_sections = split_sections(gold)
    if len(gold_sections) < min_sections:
        return None
    pred_sections = split_sections(pred)
    scores = [scorer(pred_sections.get(label, ""), body)
              for label, body in gold_sections.items()]
    return sum(scores) / len(scores) if scores else None


def section_coverage(pred: str, gold: str) -> float:
    """Fraction of gold sections the prediction even mentions.

    Reported alongside the score so "scored 0.4" can be read as either "answered
    everything mediocrely" or "answered a third of it well" -- those are very
    different failures and a single mean hides which one happened.
    """
    gold_sections = split_sections(gold)
    if not gold_sections:
        return 0.0
    pred_sections = split_sections(pred)
    return sum(1 for k in gold_sections if k in pred_sections) / len(gold_sections)
