"""Qwen3-VL-correct multi-turn SFT dataset (wired via `data.custom_cls`).

Why this exists — a REAL, MEASURED bug in verl 0.7.1's `MultiTurnSFTDataset`:

    if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        vision_position_ids = get_rope_index(...)   # verl.models.transformers.qwen2_vl

Qwen3-VL's processor exposes `image_processor` as `Qwen2VLImageProcessorFast`, so that
substring test PASSES and the **Qwen2-VL** rope is applied to a Qwen3-VL model. It does
not merely give different numbers — it raises, because Qwen3-VL's chat template
interleaves a timestamp and a `<|vision_start|>` block per temporal group while
`video_grid_thw` carries one row per video:

    IndexError: index 1 is out of bounds for dimension 0 with size 1
    (verl/models/transformers/qwen2_vl.py:119)

Measured 2026-08-17 against Qwen/Qwen3-VL-8B-Instruct with a real 4-frame echo overview
(1 video -> video_grid_thw [[2, 24, 24]], 2 `<|vision_start|>` blocks).

The RL side already does it right: `verl/utils/tokenizer.py` binds
`Qwen3VLModel.get_rope_index` onto the processor and `agent_loop.py` (v0.7.1 :745) calls
`self.processor.get_rope_index(...)`. This module makes the SFT side use that same bound
method, so cold-start SFT and the GRPO rollout build position ids identically — a
divergence there is the silent train/rollout mismatch class this project has been bitten
by twice already.

No verl patch: the shim is installed into verl's module namespace at import time, and
this module is only imported by the SFT job (through `data.custom_cls.path/name`).
"""
import verl.utils.dataset.multiturn_sft_dataset as _verl_sft
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

_verl_get_rope_index = _verl_sft.get_rope_index   # the Qwen2-VL one, kept as fallback


def model_bound_get_rope_index(
    processor,
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    second_per_grid_ts=None,
    attention_mask=None,
):
    """Drop-in replacement for `verl...qwen2_vl.get_rope_index` at the SFT call site.

    Delegates to the model's own `get_rope_index`, which `hf_processor` binds onto the
    processor per model family — Qwen2-VL, Qwen2.5-VL and Qwen3-VL all get the rope
    their own modelling code defines. Falls back to verl's Qwen2-VL implementation if
    nothing is bound.

    Shapes match the call site: unbatched `input_ids`/`attention_mask` in, (3, seq_len) out.
    `second_per_grid_ts` is accepted for signature compatibility and forwarded only to
    the fallback — Qwen3-VL dropped it (timestamps ride in `video_metadata` instead).
    """
    bound = getattr(processor, "get_rope_index", None)
    if bound is None:
        return _verl_get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

    kwargs = {}
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw
    if video_grid_thw is not None:
        kwargs["video_grid_thw"] = video_grid_thw

    vision_position_ids, _ = bound(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=None if attention_mask is None else attention_mask.unsqueeze(0),
        **kwargs,
    )
    return vision_position_ids.squeeze(1)   # (3, 1, seq_len) -> (3, seq_len)


# Installed at import time; scoped to the process that loads this dataset class.
_verl_sft.get_rope_index = model_bound_get_rope_index


class EchoMultiTurnSFTDataset(MultiTurnSFTDataset):
    """`MultiTurnSFTDataset` with model-correct mRoPE position ids for Qwen3-VL.

    Point `data.custom_cls` at this:
        data.custom_cls.path=echo_verl/sft_dataset.py
        data.custom_cls.name=EchoMultiTurnSFTDataset
    """
