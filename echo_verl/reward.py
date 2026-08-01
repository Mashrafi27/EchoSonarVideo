"""Echo reward for VeRL's custom_reward_function hook (no upstream edit).

verl calls compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs).
ground_truth carries the P1 reward_key ({kind,target,gold,is_abnormal}); we delegate
outcome+format+annealed-tool-bonus scoring to echo_rl.reward.score.total_reward.
"""
import re
from echo_rl.reward.score import total_reward

_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _count_tool_calls(solution_str: str) -> int:
    return len(_TOOLCALL_RE.findall(solution_str or ""))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    reward_key = ground_truth if isinstance(ground_truth, dict) else {"kind": "text", "target": "", "gold": {}}
    info = extra_info or {}
    result = total_reward(
        reward_key,
        solution_str or "",
        tool_calls=_count_tool_calls(solution_str),
        tool_bonus_coef=float(info.get("tool_bonus_coef", 0.0)),
    )
    return result["reward"]
