"""GREEN [Ostmeier et al., arXiv 2405.03595] -- an APPROXIMATION, not a
reproduction of EchoSonar-R's number.

EchoSonar-R computes GREEN with an "echocardiography-ADAPTED" prompt judged by
Mistral-7B (arXiv 2606.28164, Sec. 3) -- that adapted prompt was never
published, so their 0.800 (Table 3) cannot be reproduced. What CAN be
reproduced faithfully is the ORIGINAL, PUBLIC GREEN metric: its judge prompt,
error taxonomy, and scoring formula are all in the open-source reference
implementation (github.com/Stanford-AIMI/GREEN, MIT license). This module is
that prompt/parser/formula, ported to call any OpenAI-compatible judge server
instead of GREEN's own fine-tuned checkpoint -- EchoSonar-R's paper also used
Mistral-7B as judge, so pointing this at a served Mistral-7B keeps that one
detail aligned even though the prompt itself differs from theirs.

Every output of this module MUST be labeled "GREEN (ours, public prompt)" or
equivalent wherever it's displayed -- placing it unlabeled next to EchoSonar-R's
0.800 would imply a reproduction that was never possible (see nlg.py, CLAUDE.md).

Ported verbatim (prompt text, sub-category list, parsing regexes, scoring
formula) from green_score/utils.py::make_prompt and
green_score/green.py::compute_green / parse_error_counts.
"""
import re

CATEGORIES = ("Clinically Significant Errors", "Clinically Insignificant Errors",
              "Matched Findings")
SUB_CATEGORIES = (
    "(a) False report of a finding in the candidate",
    "(b) Missing a finding present in the reference",
    "(c) Misidentification of a finding's anatomic location/position",
    "(d) Misassessment of the severity of a finding",
    "(e) Mentioning a comparison that isn't in the reference",
    "(f) Omitting a comparison detailing a change from a prior study",
)


def make_prompt(reference: str, prediction: str, max_len: int = 300) -> str:
    text1 = " ".join((reference or "").split()[:max_len])
    text2 = " ".join((prediction or "").split()[:max_len])
    return (
        "Objective: Evaluate the accuracy of a candidate radiology report in "
        "comparison to a reference radiology report composed by expert "
        "radiologists.\n\n"
        "    Process Overview: You will be presented with:\n\n"
        "    1. The criteria for making a judgment.\n"
        "    2. The reference radiology report.\n"
        "    3. The candidate radiology report.\n"
        "    4. The desired format for your assessment.\n\n"
        "    1. Criteria for Judgment:\n\n"
        "    For each candidate report, determine:\n\n"
        "    The count of clinically significant errors.\n"
        "    The count of clinically insignificant errors.\n\n"
        "    Errors can fall into one of these categories:\n\n"
        "    a) False report of a finding in the candidate.\n"
        "    b) Missing a finding present in the reference.\n"
        "    c) Misidentification of a finding's anatomic location/position.\n"
        "    d) Misassessment of the severity of a finding.\n"
        "    e) Mentioning a comparison that isn't in the reference.\n"
        "    f) Omitting a comparison detailing a change from a prior study.\n"
        "    Note: Concentrate on the clinical findings rather than the "
        "report's writing style. Evaluate only the findings that appear in "
        "both reports.\n\n"
        f"    2. Reference Report:\n    {text1}\n\n"
        f"    3. Candidate Report:\n    {text2}\n\n"
        "    4. Reporting Your Assessment:\n\n"
        "    Follow this specific format for your output, even if no errors "
        "are found:\n"
        "    ```\n"
        "    [Explanation]:\n    <Explanation>\n\n"
        "    [Clinically Significant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; "
        "...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; "
        "...; <Error n>\n\n"
        "    [Clinically Insignificant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; "
        "...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; "
        "...; <Error n>\n\n"
        "    [Matched Findings]:\n"
        "    <The number of matched findings>. <Finding 1>; <Finding 2>; "
        "...; <Finding n>\n"
        "    ```\n"
    )


def _parse_error_counts(text: str, category: str):
    """-> (total, [6 sub-category counts]). Matches Stanford-AIMI/GREEN's
    parse_error_counts for the two branches this module needs (Matched
    Findings is a single count, error categories are 6 lettered sub-counts)."""
    pattern = rf"\[{re.escape(category)}\]:\s*(.*?)(?:\n\s*\n|\Z)"
    m = re.search(pattern, text or "", re.DOTALL)
    if not m or m.group(1).strip().startswith("No"):
        return 0, [0] * 6
    body = m.group(1)
    if category == "Matched Findings":
        counts = re.findall(r"^\s*\b\d+\b(?=\.)", body)
        return (int(counts[0]) if counts else 0), [0] * 6
    sub_prefixes = [s.split(" ", 1)[0] + " " for s in SUB_CATEGORIES]
    matches = sorted(re.findall(r"\([a-f]\) .*", body))
    sub_counts = [0] * 6
    for pos, prefix in enumerate(sub_prefixes):
        for match in matches:
            if match.startswith(prefix):
                count = re.findall(r"(?<=: )\b\d+\b(?=\.)", match)
                if count:
                    sub_counts[pos] = int(count[0])
    return sum(sub_counts), sub_counts


def score_one(judge_response: str) -> dict:
    """One judged report -> matched findings, the 6 significant-error
    sub-counts, and the GREEN score (matched / (matched + sig_errors), 0 if
    nothing matched -- ported from GREEN.compute_green)."""
    matched, _ = _parse_error_counts(judge_response, "Matched Findings")
    sig_total, sig_errors = _parse_error_counts(judge_response, "Clinically Significant Errors")
    green = 0.0 if matched == 0 else matched / (matched + sig_total)
    return {"matched_findings": matched, "sig_errors": sig_errors,
            "sig_errors_total": sig_total, "green": green}


def green_score_corpus(client, model: str, predictions: list, references: list, *,
                        temperature: float = 0.0, max_tokens: int = 1024) -> dict:
    """Judge every (prediction, reference) pair with the served `model`
    (OpenAI-compatible `client`, e.g. a Mistral-7B vLLM server) and return
    per-example scores plus the corpus mean. Key is deliberately
    `green_ours_public_prompt`, never `green`, so it can't be mistaken for
    EchoSonar-R's number downstream."""
    per_example = []
    for pred, ref in zip(predictions, references):
        prompt = make_prompt(ref, pred)
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content or ""
        per_example.append({**score_one(text), "judge_response": text})
    scores = [r["green"] for r in per_example]
    return {"green_ours_public_prompt": sum(scores) / len(scores) if scores else 0.0,
            "n": len(per_example), "per_example": per_example}
