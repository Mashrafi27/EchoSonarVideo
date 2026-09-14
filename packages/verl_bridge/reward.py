"""Echo reward for VeRL's custom_reward_function hook (no upstream edit).

verl calls compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs).
ground_truth carries the P1 reward_key ({kind,target,gold,is_abnormal}); we delegate
outcome+format+annealed-tool-bonus scoring to data_core.reward.score.total_reward.
"""
import json
import os
import re
import time
from data_core.reward.score import total_reward

_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)

_DEFAULT_KEY = {"kind": "text", "target": "", "gold": {}}

# Per-process episode counter + wall clock, so we can see live rollout throughput
# in the driver log instead of waiting blind on a whole batch. One RewardLoopWorker
# actor per process, so this counts "episodes finished by THIS worker", not global.
_EPISODE_COUNT = 0
_START_T = time.monotonic()


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
    n_tool_calls = _count_tool_calls(solution_str)
    result = total_reward(
        reward_key,
        solution_str or "",
        tool_calls=n_tool_calls,
        tool_bonus_coef=float(info.get("tool_bonus_coef", 0.0)),
    )
    global _EPISODE_COUNT
    _EPISODE_COUNT += 1
    elapsed = time.monotonic() - _START_T
    print(f"[reward pid={os.getpid()}] episode #{_EPISODE_COUNT} finished "
          f"reward={result['reward']:.3f} tool_calls={n_tool_calls} "
          f"chars={len(solution_str or '')} elapsed={elapsed:.1f}s", flush=True)
    return result["reward"]
