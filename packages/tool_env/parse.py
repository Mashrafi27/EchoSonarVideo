import re
import json
from dataclasses import dataclass, field

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ParsedAction:
    answer: str | None = None
    calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def parse_action(action_string: str) -> ParsedAction:
    result = ParsedAction()
    answers = _ANSWER_RE.findall(action_string or "")
    if answers:
        result.answer = answers[-1].strip()
    for raw in _TOOL_RE.findall(action_string or ""):
        try:
            obj = json.loads(raw.strip())
        except (ValueError, TypeError) as e:
            result.errors.append(f"malformed tool_call JSON: {e}")
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            result.errors.append(f"tool_call missing 'name': {raw.strip()[:80]}")
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        result.calls.append({"name": obj["name"], "arguments": args})
    return result
