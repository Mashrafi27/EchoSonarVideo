"""Echo reward for VeRL's custom_reward_function hook (no upstream edit).

verl calls compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs).
ground_truth carries the P1 reward_key ({kind,target,gold,is_abnormal}); we delegate
outcome+format+annealed-tool-bonus scoring to echo_rl.reward.score.total_reward.
"""
import json
import re
from echo_rl.reward.score import total_reward

_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)

_DEFAULT_KEY = {"kind": "text", "target": "", "gold": {}}


def _parse_reward_key(ground_truth):
    # ground_truth may be a dict (direct/test call) or a JSON string (parquet column).
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            obj = json.loads(ground_truth)
        except (ValueError, TypeError):
            return _DEFAULT_KEY
        return obj if isinstance(obj, dict) else _DEFAULT_KEY
    return _DEFAULT_KEY


def _count_tool_calls(solution_str: str) -> int:
    return len(_TOOLCALL_RE.findall(solution_str or ""))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    reward_key = _parse_reward_key(ground_truth)
    info = extra_info or {}
    result = total_reward(
        reward_key,
        solution_str or "",
        tool_calls=_count_tool_calls(solution_str),
        tool_bonus_coef=float(info.get("tool_bonus_coef", 0.0)),
    )
    return result["reward"]
