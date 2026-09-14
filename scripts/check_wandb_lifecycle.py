"""CPU-only Ray regression: flush W&B before actor teardown, including failures.

Uses only offline synthetic diagnostic runs; never publishes test metrics.
"""
import json
import os
from pathlib import Path

import ray


@ray.remote(num_cpus=1)
class Probe:
    def run(self, root, fail):
        import uvloop
        uvloop.install()
        from verl.utils.tracking import Tracking
        import wandb
        from verl_bridge.wandb_lifecycle import run_with_wandb_finish

        directory = Path(root) / ('failure' if fail else 'success')
        directory.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_DIR'] = str(directory)
        os.environ['VERL_FILE_LOGGER_PATH'] = str(directory / 'metrics.jsonl')
        original_run = None

        def task():
            nonlocal original_run
            # Retain the tracker like the original delayed __del__ case.
            self.tracker = Tracking('echo-wandb-lifecycle-offline', 'synthetic-diagnostic',
                                    ['console', 'file', 'wandb'])
            original_run = wandb.run
            for step in (1, 2):
                self.tracker.log({'diagnostic/value': float(step)}, step=step)
            if fail:
                raise ValueError('intentional diagnostic failure')
            return 'completed'

        try:
            outcome = run_with_wandb_finish(task)
            assert not fail and outcome == 'completed'
        except ValueError as exc:
            assert fail and str(exc) == 'intentional diagnostic failure'
        assert wandb.run is None, 'W&B still active when Ray worker returns'
        assert original_run._is_finished, 'W&B finish did not complete'
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == [1, 2], rows
        self.tracker.__del__()
        self.tracker.logger.clear()
        return {'failure_case': fail, 'flushed': True, 'steps': [1, 2]}


def main():
    root = Path(os.environ['WANDB_LIFECYCLE_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=83886080,
             _temp_dir=os.environ['RAY_TMPDIR'], _plasma_directory=os.environ['TMPDIR'])
    results = []
    try:
        for fail in (False, True):
            actor = Probe.remote()
            results.append(ray.get(actor.run.remote(str(root), fail)))
            ray.kill(actor)
    finally:
        ray.shutdown()
    (root / 'result.json').write_text(json.dumps({'pass': True, 'cases': results}, indent=2)+'\n')
    print('WANDB_LIFECYCLE_PASS', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
