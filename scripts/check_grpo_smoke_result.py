"""Summarize real VeRL step logs; reject startup-only or zero-gradient runs."""
import argparse
import json
import math
from pathlib import Path
import re

ap = argparse.ArgumentParser()
ap.add_argument('--run-root', required=True, type=Path)
ap.add_argument('--job-id', required=True)
args = ap.parse_args()
log = args.run_root / f'grpo_{args.job_id}.out'
text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace'))
steps = {}
for line in text.splitlines():
    match = re.search(r'\bstep:(\d+) - ', line)
    if not match:
        continue
    metrics = {}
    for item in line[match.end():].split(' - '):
        key, sep, value = item.partition(':')
        if sep:
            try:
                value = value.strip()
                wrapped = re.fullmatch(r'np\.(?:float|int|uint)\d+\(([^()]*)\)', value)
                metrics[key] = float(wrapped[1] if wrapped else value)
            except ValueError:
                pass
    steps[int(match[1])] = metrics
required = ('actor/grad_norm', 'actor/pg_loss', 'actor/kl_loss',
            'critic/advantages/min', 'critic/advantages/max',
            'timing_s/update_actor', 'timing_s/update_weights',
            'actor/vision_position_max_update')
issues = []
for step in (1, 2):
    metrics = steps.get(step, {})
    for key in required:
        if key not in metrics or not math.isfinite(metrics[key]):
            issues.append(f'Step {step}: missing/non-finite {key}')
    for key in ('timing_s/update_actor', 'timing_s/update_weights'):
        if metrics.get(key, 0) <= 0:
            issues.append(f'Step {step}: no positive {key}')
if not any(m.get('actor/grad_norm', 0) > 0 for m in steps.values()):
    issues.append('No nonzero actor gradient')
if not any(m.get('critic/advantages/max', 0) > m.get('critic/advantages/min', 0) for m in steps.values()):
    issues.append('No within-group reward advantage variation')
if not any(m.get('actor/vision_position_max_update', 0) > 0 for m in steps.values()):
    issues.append('No measured change to vision position weights')
out = args.run_root / f'grpo_{args.job_id}'
out.mkdir(exist_ok=True)
result = {'job_id': args.job_id, 'pass': not issues, 'issues': issues,
          'steps': steps, 'scope': 'Two-step infrastructure smoke; no clinical-quality claim'}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'job_id': args.job_id, 'pass': not issues, 'issues': issues,
                  'completed_steps': sorted(steps)}))
raise SystemExit(bool(issues))
