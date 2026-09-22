"""Gate a full launch on two completed diagnostic updates and actual rollout content."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from echoprime_track.prompts import SYSTEM_PROMPT


def check_run(run_dir):
    root = Path(run_dir)
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
               if line.strip()]
    if [row["step"] for row in metrics] != [1, 2]:
        raise ValueError("Expected exactly two completed diagnostic updates")
    gradients, clip_ratios = [], []
    rows = []
    varying_groups = 0
    for metric in metrics:
        data = metric["data"]
        grad = float(data["actor/grad_norm"])
        clip = float(data["response_length/clip_ratio"])
        if not math.isfinite(grad) or grad < 0 or not 0 <= clip <= 1:
            raise ValueError("Invalid gradient or truncation metric")
        if not float(data["timing_s/update_weights"]) > 0:
            raise ValueError("Missing completed weight sync")
        gradients.append(grad)
        clip_ratios.append(clip)
        step_rows = [json.loads(line) for line in
                     (root / "rollouts" / f'{metric["step"]}.jsonl').read_text().splitlines()
                     if line.strip()]
        if len(step_rows) != 8:
            raise ValueError("Expected eight responses per diagnostic update")
        groups = defaultdict(list)
        for row in step_rows:
            if SYSTEM_PROMPT not in row["input"] or not row["input"].rstrip().endswith("<think>"):
                raise ValueError("Actual rollout input lacks the canonical prompt or priming")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("Nonfinite reward")
            groups[row["input"]].append(score)
        if len(groups) != 4 or any(len(scores) != 2 for scores in groups.values()):
            raise ValueError("Unexpected diagnostic prompt grouping")
        varying_groups += sum(len(set(scores)) > 1 for scores in groups.values())
        rows.extend(step_rows)
    closed = sum("</think>" in row["output"] for row in rows)
    positive = sum(float(row["score"]) > 0 for row in rows)
    tool_calls = sum(row["output"].count("<tool_call>") for row in rows)
    tool_results = sum(row["output"].count("<tool_response>") for row in rows)
    if not any(g > 0 for g in gradients) or not varying_groups or not positive:
        raise ValueError("No usable within-group GRPO learning signal")
    if closed < len(rows) / 2 or sum(clip_ratios) / len(clip_ratios) > 0.25:
        raise ValueError("Too few completed thinking blocks or excessive truncation")
    if tool_calls and not tool_results:
        raise ValueError("Tool calls occurred but no tool result reached the trajectory")
    return {"gate": "PASS", "updates": len(metrics), "responses": len(rows),
            "positive_rewards": positive, "closed_thinking": closed,
            "gradients": gradients, "clip_ratios": clip_ratios,
            "groups_with_reward_variation": varying_groups,
            "tool_calls": tool_calls, "tool_results": tool_results,
            "policy_tool_use_observed": bool(tool_results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(check_run(args.run_dir)), flush=True)


if __name__ == "__main__":
    main()
