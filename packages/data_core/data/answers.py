import json
import re

_NORMAL_HINT = re.compile(r"\bno (significant )?abnormalit", re.I)


def last_answer(messages_field) -> str:
    msgs = json.loads(messages_field) if isinstance(messages_field, str) else messages_field
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m["content"].strip()
    return ""


def parse_yes_no(answer: str):
    tok = answer.strip().lower().lstrip(".,! ")[:4]
    if tok.startswith("yes") and (len(tok) <= 3 or not tok[3].isalpha()):
        return "yes"
    if tok.startswith("no") and (len(tok) <= 2 or not tok[2].isalpha()):
        return "no"
    return None


def finding_set(list_answer: str) -> set:
    if _NORMAL_HINT.search(list_answer):
        return set()
    out = set()
    for line in list_answer.splitlines():
        line = line.strip()
        if line.startswith("-"):
            name = line.lstrip("- ").strip().lower()
            if name:
                out.add(name)
    return out


def is_abnormal(question_type: str, answer: str) -> bool:
    if question_type == "abnormality_classification":
        return parse_yes_no(answer) == "yes"
    if question_type == "abnormality_list":
        return len(finding_set(answer)) > 0
    a = answer.lower()
    if _NORMAL_HINT.search(a):
        return False
    return bool(re.search(r"(dilat|reduced|abnormal|severe|moderate|regurgitat|stenos|hypertroph|impaired|akinet|hypokinet)", a))
