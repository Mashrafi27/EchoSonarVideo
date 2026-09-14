"""Qwen3-VL FSDP2 wrapping compatibility, loaded via model.external_lib."""
from functools import wraps
from verl.utils import fsdp_utils


def install():
    original = fsdp_utils._select_fsdp2_wrap_targets
    if getattr(original, '_echo_vision_position_fix', False):
        return

    @wraps(original)
    def select(model, transformer_classes):
        targets = original(model, transformer_classes)
        if getattr(model.config, 'model_type', None) != 'qwen3_vl':
            return targets
        visual = getattr(getattr(model, 'model', None), 'visual', None)
        position_embedding = getattr(visual, 'pos_embed', None)
        # Qwen reads this weight's device before invoking the embedding. If the
        # embedding is a separate offloaded FSDP unit, that device is still CPU.
        # Keep this small parameter in the root FSDP unit, whose pre-forward
        # gather runs before Qwen constructs image-position indices.
        return [module for module in targets if module is not position_embedding]

    select._echo_vision_position_fix = True
    fsdp_utils._select_fsdp2_wrap_targets = select


install()

# The opt-in smoke extension also reports stages before a whole step completes.
from verl_bridge.training_progress import install as install_progress
install_progress()
