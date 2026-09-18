"""Pure, model-free reward scoring over Phase-1 reward_keys.

Outcome scorers return a float in [0, 1]. The LLM-judge is injected behind the
JudgeClient interface; NullJudge (offline) returns None so score_text falls back
to the clinical-entity-F1 co-signal. A real vLLM judge client is P3e.

Reward formula (multi-turn tool track, matching EchoSonar-R's r_fmt gating):

    r_fmt  = 1 if the response has at least one </think> and n_close == n_open+1
             (the opening <think> is primed in the prompt, not in the response)
             AND all <tool_call> blocks are valid JSON with {name, arguments}, else 0
    r_cor  = score_outcome(...)        # 0/1 for yesno, IoU∈[0,1] for set
    r_tool = tool_bonus_coef × r_cor  if tool_calls >= 1, else 0
    r_len  = min(0, (L - L_min) / L_min)  where L_min = 200 × (tool_calls + 1)
             -- overshort penalty scaled by turns taken, never a positive bonus

    reward = r_fmt × (r_cor + r_tool) + r_len
"""
import json as _json
import re

from data_core.reward.findings import extract_canonical_findings, iou
from data_core.reward.sections import score_by_section
from data_core.data.answers import parse_yes_no

# Clinical-finding vocabulary for free-text entity extraction (mirrors
# data_core.data.answers.is_abnormal's abnormal-keyword set).
_ENTITY_RE = re.compile(
    r"(dilat|reduced|abnormal|severe|moderate|mild|regurgitat|stenos|"
    r"hypertroph|impaired|akinet|hypokinet|effusion|thromb|normal)", re.I)

_L_MIN_PER_TURN = 200   # Darya's single-turn minimum; we scale by turns taken


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_yesno(pred_answer: str, target: str) -> float:
    return 1.0 if target is not None and parse_yes_no(pred_answer or "") == target else 0.0


def score_set(pred_answer: str, target: list) -> float:
    return iou(extract_canonical_findings(pred_answer or ""), set(target or []))


def extract_entities(text: str) -> set:
    return {m.group(1).lower() for m in _ENTITY_RE.finditer(text or "")}


def score_entity_f1(pred_answer: str, ref_answer: str) -> float:
    return f1(extract_entities(pred_answer), extract_entities(ref_answer))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def score_gold_value(pred_answer: str, gold: dict) -> float | None:
    labels = [v for v in (gold or {}).values() if v]
    if not labels:
        return None
    p = _norm(pred_answer)
    hits = sum(1 for lab in labels if _norm(lab) in p)
    return hits / len(labels)


class JudgeClient:
    def score(self, question: str, pred: str, ref: str):
        raise NotImplementedError


class NullJudge(JudgeClient):
    def score(self, question: str, pred: str, ref: str):
        return None


def _score_text(pred: str, ref: str, question: str, judge: JudgeClient) -> float:
    """Score one free-text span: judge blended with entity-F1, or entity-F1 alone."""
    jv = judge.score(question, pred, ref)
    ef = score_entity_f1(pred, ref)
    return 0.5 * jv + 0.5 * ef if jv is not None else ef


def score_outcome(reward_key: dict, pred_answer: str, *, question: str = "",
                  judge: JudgeClient = NullJudge()) -> float:
    kind = reward_key.get("kind")
    if kind == "yesno":
        return score_yesno(pred_answer, reward_key.get("target"))
    if kind == "set":
        return score_set(pred_answer, reward_key.get("target"))
    # text: prefer structured gold; else per-section; else whole-text.
    gv = score_gold_value(pred_answer, reward_key.get("gold"))
    if gv is not None:
        return gv
    ref = reward_key.get("target", "")
    # A full_report carries a median of 13 labelled sections. Scoring it as one
    # blob makes "LV right, RV wrong" indistinguishable from the reverse and gives
    # no gradient toward fixing either. Score per section and average, as
    # EchoSonar-R does (arXiv 2606.28164 eq. r_cor^report). Falls through to
    # whole-text scoring when the gold has no section structure.
    sectioned = score_by_section(
        pred_answer, ref, lambda p, g: _score_text(p, g, question, judge))
    if sectioned is not None:
        return sectioned
    return _score_text(pred_answer, ref, question, judge)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
_THINK_OPEN_RE = re.compile(r"<think>", re.S)
_THINK_CLOSE_RE = re.compile(r"</think>", re.S)
_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_TOOLCALL_STRIP_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def extract_answer(completion: str) -> str | None:
    """Extract the final answer from a (possibly multi-turn) completion.

    Prefers explicit <answer> tags (Qwen3-VL tool track).  Falls back to
    text after the last </think>, stripping any trailing tool_call blocks
    (Darya's SFT format and our multi-turn echoprime track).
    """
    matches = _ANSWER_RE.findall(completion or "")
    if matches:
        return matches[-1].strip() or None
    # Fallback: everything after the last </think>, minus any tool calls.
    parts = (completion or "").rsplit("</think>", 1)
    if len(parts) == 2:
        tail = _TOOLCALL_STRIP_RE.sub("", parts[1]).strip()
        return tail or None
    return None


def _all_tool_calls_valid(completion: str) -> bool:
    """Return True iff every <tool_call> block is valid JSON with name+arguments."""
    bodies = _TOOLCALL_RE.findall(completion or "")
    if not bodies:
        return True  # no tool calls is fine
    for body in bodies:
        try:
            payload = _json.loads(body.strip())
        except (ValueError, TypeError):
            return False
        if not (isinstance(payload, dict)
                and "name" in payload
                and "arguments" in payload):
            return False
    return True


def score_format(completion: str) -> float:
    """r_fmt: 1 iff the multi-turn response is structurally correct.

    Checks:
    1. At least one <think>...</think> block present.
    2. Opening and closing think tags are balanced (one per turn).
    3. All <tool_call> blocks are valid JSON with {name, arguments}.
    """
    text = completion or ""
    n_open = len(_THINK_OPEN_RE.findall(text))
    n_close = len(_THINK_CLOSE_RE.findall(text))
    # Allow n_close == n_open + 1: the opening <think> is primed in the prompt
    # (never in the response), so single-turn responses contain </think> but not
    # <think>. Multi-turn: each tool round adds one <think> in the response, so
    # n_close is always n_open + 1 (the primed one). Require at least one </think>.
    if n_close == 0 or abs(n_open - n_close) > 1:
        return 0.0
    if not _all_tool_calls_valid(text):
        return 0.0
    return 1.0


def total_reward(reward_key: dict, completion: str, *, tool_calls: int = 0,
                 tool_bonus_coef: float = 0.0, question: str = "",
                 judge: JudgeClient = NullJudge()) -> dict:
    """Compute reward using EchoSonar-R's multiplicative format gating + tool bonus.

        reward = r_fmt × (r_cor + r_tool) + r_len

    r_fmt gates both outcome and tool bonus: wrong format → zero reward.
    r_tool = tool_bonus_coef × r_cor when tool_calls >= 1 (partial for set IoU).
    r_len  = min(0, (L - L_min) / L_min), L_min = 200 × (tool_calls + 1).
    """
    answer = extract_answer(completion)
    r_cor = score_outcome(reward_key, answer, question=question, judge=judge) if answer else 0.0
    r_fmt = score_format(completion)
    r_tool = tool_bonus_coef * r_cor if tool_calls >= 1 else 0.0

    n_tokens = len((completion or "").split())  # rough token count
    l_min = _L_MIN_PER_TURN * (tool_calls + 1)
    r_len = min(0.0, (n_tokens - l_min) / l_min) if l_min > 0 else 0.0

    reward = r_fmt * (r_cor + r_tool) + r_len
    return {"reward": reward, "outcome": r_cor, "format": r_fmt,
            "tool_bonus": r_tool, "length_penalty": r_len}
