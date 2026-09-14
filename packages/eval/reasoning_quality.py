"""Table 2 (EchoSonar-R, arXiv 2606.28164): reasoning-trace quality via
LLM-as-judge, 1-5 Likert across five dimensions.

EchoSonar-R's exact judge prompt was never published -- only the five
dimension DEFINITIONS are (Sec. 3, "Evaluation Metrics", quoted verbatim in
DIMENSIONS below). This module writes OUR OWN judge prompt from those
published definitions and runs it against whichever judge model is served
(the paper uses Mistral-7B). Every number this produces is "our
interpretation of their rubric," not a reproduction of their Table 2 -- label
it that way wherever it's shown, the same way green_approx.py's GREEN is
labeled "ours, public prompt".
"""
import json
import re

DIMENSIONS = {
    "reasoning_answer_agreement": (
        "Reasoning-Answer Agreement",
        "Whether every abnormality discussed in the reasoning trace is "
        "reflected in the final answer, and vice versa."),
    "reasoning_efficiency": (
        "Reasoning Efficiency",
        "Whether the reasoning is concise and focused, without redundant or "
        "circular steps."),
    "factual_correctness": (
        "Factual Correctness",
        "The accuracy of clinical and anatomical claims made during "
        "reasoning."),
    "evidence_grounding": (
        "Evidence Grounding",
        "Whether the model references specific visual observations from the "
        "echocardiographic input rather than relying on generic statements."),
    "terminology_accuracy": (
        "Terminology Accuracy",
        "The correct use of echocardiographic and medical terminology "
        "throughout the reasoning chain."),
}

_RUBRIC = "\n".join(f"- {key} ({name}): {desc}"
                     for key, (name, desc) in DIMENSIONS.items())


def make_prompt(question: str, reasoning_trace: str, final_answer: str) -> str:
    return (
        "You are an expert echocardiographer grading the REASONING TRACE of "
        "an AI assistant that answered a question about a cardiac ultrasound "
        "study.\n\n"
        "Score the reasoning trace on a 1-5 Likert scale (5 = best, 1 = "
        f"worst) across these five dimensions:\n{_RUBRIC}\n\n"
        f"Question: {question}\n\n"
        f"Reasoning trace:\n{reasoning_trace}\n\n"
        f"Final answer: {final_answer}\n\n"
        "Respond with ONLY a JSON object mapping each dimension's key above "
        "to an integer 1-5, e.g.:\n"
        '{"reasoning_answer_agreement": 5, "reasoning_efficiency": 4, '
        '"factual_correctness": 5, "evidence_grounding": 3, '
        '"terminology_accuracy": 5}'
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_scores(judge_response: str) -> dict:
    """Judge text -> {dimension: int|None}. None where the judge didn't
    return a parseable 1-5 integer for that dimension -- a missing score is
    dropped from that dimension's average, not counted as 0 (see
    reasoning_quality_corpus)."""
    out = {k: None for k in DIMENSIONS}
    m = _JSON_BLOCK.search(judge_response or "")
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for k in DIMENSIONS:
        v = data.get(k)
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            out[k] = int(v)
    return out


def reasoning_quality_corpus(client, model: str, records: list, *,
                              temperature: float = 0.0, max_tokens: int = 512) -> dict:
    """`records`: list of {question, reasoning_trace, final_answer}.

    Returns per-dimension means (over examples where that dimension parsed)
    plus an overall average across the five dimension means -- matching
    Table 2's shape (rows = dimensions, plus an Average row).
    """
    per_example = []
    for r in records:
        prompt = make_prompt(r["question"], r["reasoning_trace"], r["final_answer"])
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content or ""
        per_example.append({**parse_scores(text), "judge_response": text})

    means = {}
    for k in DIMENSIONS:
        vals = [e[k] for e in per_example if e[k] is not None]
        means[k] = sum(vals) / len(vals) if vals else None
    scored = [v for v in means.values() if v is not None]
    means["average"] = sum(scored) / len(scored) if scored else None
    return {"means": means, "n": len(per_example),
            "n_fully_unparsed": sum(1 for e in per_example
                                     if all(e[k] is None for k in DIMENSIONS)),
            "per_example": per_example}
