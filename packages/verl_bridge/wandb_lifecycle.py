"""Flush W&B inside the training worker, before Ray can tear that worker down."""
from __future__ import annotations

import sys


def _finish(exit_code: int) -> None:
    wandb = sys.modules.get('wandb')
    run = getattr(wandb, 'run', None)
    if run is None:
        return
    print(f'ECHO_WANDB_FINISH_START run={run.id} exit_code={exit_code}', flush=True)
    run.finish(exit_code=exit_code)
    print(f'ECHO_WANDB_FINISH_COMPLETE run={run.id} exit_code={exit_code}', flush=True)


def run_with_wandb_finish(task, *args, **kwargs):
    """Preserve training failures and surface flush failures on successful runs."""
    try:
        result = task(*args, **kwargs)
    except BaseException:
        try:
            _finish(exit_code=1)
        except Exception as exc:
            print(f'ECHO_WANDB_FINISH_FAILED during training failure: {exc}', flush=True)
        raise
    _finish(exit_code=0)
    return result
