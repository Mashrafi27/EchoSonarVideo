"""Opt-in stage markers and actual parameter-change evidence for the smoke run."""
from functools import wraps
import time
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor


def install():
    for name in ('compute_log_prob', 'update_policy'):
        original = getattr(DataParallelPPOActor, name)
        if getattr(original, '_echo_progress', False):
            continue

        def make_wrapper(method, method_name):
            @wraps(method)
            def run(self, *args, **kwargs):
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                phase = method_name
                if method_name == 'compute_log_prob':
                    phase = 'actor_log_prob' if kwargs.get('calculate_entropy') else 'reference_log_prob'
                if rank == 0:
                    print(f'ECHO_GRPO_PHASE phase={phase} event=start', flush=True)
                started = time.monotonic()
                before = None
                if method_name == 'update_policy':
                    visual = getattr(getattr(self.actor_module, 'model', None), 'visual', None)
                    position = getattr(visual, 'pos_embed', None)
                    if position is not None:
                        weight = position.weight
                        local = weight.to_local() if hasattr(weight, 'to_local') else weight
                        before = local.detach().cpu().clone()
                try:
                    result = method(self, *args, **kwargs)
                except Exception:
                    if rank == 0:
                        print(f'ECHO_GRPO_PHASE phase={phase} event=failed elapsed_s={time.monotonic()-started:.2f}', flush=True)
                    raise
                if before is not None:
                    weight = position.weight
                    local = weight.to_local() if hasattr(weight, 'to_local') else weight
                    delta = (local.detach().cpu() - before).abs()
                    result['actor/vision_position_max_update'] = delta.max().item()
                    result['actor/vision_position_changed_elements'] = int(torch.count_nonzero(delta))
                if rank == 0:
                    print(f'ECHO_GRPO_PHASE phase={phase} event=complete elapsed_s={time.monotonic()-started:.2f}', flush=True)
                return result
            run._echo_progress = True
            return run

        setattr(DataParallelPPOActor, name, make_wrapper(original, name))
