"""Pure, model-free reward scoring over Phase-1 reward_keys.

Outcome scorers return a float in [0, 1]. The LLM-judge is injected behind the
JudgeClient interface; NullJudge (offline) returns None so score_text falls back
to the clinical-entity-F1 co-signal. A real vLLM judge client is P3e.
"""
import re
from echo_rl.data.answers import parse_yes_no, finding_set

# Clinical-finding vocabulary for free-text entity extraction (mirrors
# echo_rl.data.answers.is_abnormal's abnormal-keyword set).
_ENTITY_RE = re.compile(
    r"(dilat|reduced|abnormal|severe|moderate|mild|regurgitat|stenos|"
    r"hypertroph|impaired|akinet|hypokinet|effusion|thromb|normal)", re.I)


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
    return 1.0 if parse_yes_no(pred_answer) == target else 0.0


def score_set(pred_answer: str, target: list) -> float:
    return f1(finding_set(pred_answer), set(target or []))


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


def score_outcome(reward_key: dict, pred_answer: str, *, question: str = "",
                  judge: JudgeClient = NullJudge()) -> float:
    kind = reward_key.get("kind")
    if kind == "yesno":
        return score_yesno(pred_answer, reward_key.get("target"))
    if kind == "set":
        return score_set(pred_answer, reward_key.get("target"))
    # text: prefer structured gold; else judge (blended with entity-F1); else entity-F1.
    gv = score_gold_value(pred_answer, reward_key.get("gold"))
    if gv is not None:
        return gv
    ref = reward_key.get("target", "")
    jv = judge.score(question, pred_answer, ref)
    ef = score_entity_f1(pred_answer, ref)
    if jv is not None:
        return 0.5 * jv + 0.5 * ef
    return ef
