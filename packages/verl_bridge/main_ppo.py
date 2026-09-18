"""Upstream PPO entry point with explicit W&B finalization in the Ray worker."""
from pathlib import Path

import hydra
import ray
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer import main_ppo as upstream
from verl.utils.device import auto_set_device

from verl_bridge.wandb_lifecycle import run_with_wandb_finish


class EchoTaskRunner(upstream.TaskRunner):
    def run(self, config):
        try:
            import weave
            weave.init(config.trainer.project_name)
        except Exception as exc:
            print(f"ECHO_WEAVE_INIT_FAILED: {exc}", flush=True)
        return run_with_wandb_finish(super().run, config)


@hydra.main(
    config_path=str(Path(upstream.__file__).parent / 'config'),
    config_name='ppo_trainer',
    version_base=None,
)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    upstream.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(EchoTaskRunner))


if __name__ == '__main__':
    main()
