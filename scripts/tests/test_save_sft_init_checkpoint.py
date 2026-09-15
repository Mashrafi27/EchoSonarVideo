import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from save_sft_init_checkpoint import load_darya_projectors


def _toy_projector(in_dim, out_dim):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))


def test_load_darya_projectors_copies_real_weights(tmp_path):
    # Simulate EchoVLM.load_projectors's save format:
    # <save_dir>/clip_projector.pt, <save_dir>/detr_projector.pt
    src_clip = _toy_projector(768, 16)
    src_detr = _toy_projector(256, 16)
    torch.save(src_clip.state_dict(), tmp_path / "clip_projector.pt")
    torch.save(src_detr.state_dict(), tmp_path / "detr_projector.pt")

    dst_clip = _toy_projector(768, 16)
    dst_detr = _toy_projector(256, 16)
    # dst starts randomly initialized -- must differ from src before loading
    assert not torch.equal(dst_clip[1].weight, src_clip[1].weight)

    load_darya_projectors(dst_clip, dst_detr, str(tmp_path))

    assert torch.equal(dst_clip[1].weight, src_clip[1].weight)
    assert torch.equal(dst_detr[1].weight, src_detr[1].weight)


def test_load_darya_projectors_missing_file_raises(tmp_path):
    dst_clip = _toy_projector(768, 16)
    dst_detr = _toy_projector(256, 16)
    try:
        load_darya_projectors(dst_clip, dst_detr, str(tmp_path))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
