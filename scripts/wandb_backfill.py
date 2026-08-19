#!/usr/bin/env python3
"""Replay a finished verl console log into a wandb run.

    python scripts/wandb_backfill.py logs/echo_sft_141969.out \
        --project echo-sft --name coldstart-qwen3vl8b-141969

Why this exists: the first full SFT run (job 141969) trained for 2h55m with
trainer.logger=['console'], so its curves live only in a text file. Rather than
spend another three hours of MI210 time to get a chart, replay the log.

verl's console logger emits one line per step:

    step:1 - train/loss:1.36... - train/grad_norm:19.875 - train/lr:1.66e-06 - ...

which is lossless for scalars, so the backfilled run carries the same numbers a
live wandb run would have. It CANNOT carry what was never printed: system metrics
(GPU utilisation, power), gradients/parameter histograms, or wall-clock timestamps.
Backfilled runs are therefore tagged `backfill` and get `backfilled_from` in their
config, so nobody later reads a missing GPU chart as an outage.

Use --dry-run to check parsing without touching the network.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# "step:12" then any number of " - key:value" pairs. Values that are not numeric
# (verl prints a few strings) are skipped rather than coerced.
_STEP = re.compile(r"^step:(\d+)\s*(.*)$")
_PAIR = re.compile(r"([A-Za-z0-9_/()\.]+):([^\s-][^\s]*|-?\d[\d.eE+-]*)")


def parse_log(path: Path):
    """-> list of (step, {metric: float}). Non-numeric metrics are dropped."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = _STEP.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        metrics = {}
        for key, raw in _PAIR.findall(m.group(2)):
            try:
                metrics[key] = float(raw)
            except ValueError:
                continue
        if metrics:
            rows.append((step, metrics))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--project", default="echo-sft")
    ap.add_argument("--name", default=None, help="wandb run name (default: log stem)")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, log nothing")
    args = ap.parse_args(argv)

    if not args.log.exists():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 1

    rows = parse_log(args.log)
    if not rows:
        print(f"no 'step:N - k:v' lines found in {args.log}", file=sys.stderr)
        return 1

    keys = sorted({k for _, m in rows for k in m})
    first, last = rows[0], rows[-1]
    print(f"parsed {len(rows)} steps ({first[0]}..{last[0]}) from {args.log}")
    print(f"metrics: {', '.join(keys)}")
    for label, (step, metrics) in (("first", first), ("last", last)):
        shown = {k: metrics[k] for k in ("train/loss", "train/grad_norm") if k in metrics}
        print(f"  {label} step {step}: {shown}")
    if args.dry_run:
        return 0

    import wandb
    run = wandb.init(
        project=args.project,
        name=args.name or args.log.stem,
        tags=["backfill"],
        # Recorded so a reader can tell why the system-metrics panels are empty:
        # nothing was sampled live, this is a replay of stdout.
        config={"backfilled_from": str(args.log), "backfill_steps": len(rows)},
    )
    for step, metrics in rows:
        run.log(metrics, step=step)
    url = run.url
    run.finish()
    print(f"backfilled {len(rows)} steps -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
