"""Regression: the Qwen3-VL rope shim is load-bearing, in both directions.

Heavy — needs the training venv (`.venv-train`), a real verl install and the
17 GB Qwen3-VL checkpoint in the HF cache. Skipped everywhere else, so the
offline `.venv` suite stays fast:

    .venv-train/bin/python -m pytest echo_verl/tests/test_sft_dataset_rope.py

Builds an 8-row SFT parquet from build/sft.jsonl if one isn't already there.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("verl")
pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[2]
_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_PARQUET = _REPO / "build" / "sft_smoke.parquet"
_SFT_JSONL = _REPO / "build" / "sft.jsonl"

if not (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct").exists():
    pytest.skip("Qwen3-VL-8B-Instruct not in the HF cache", allow_module_level=True)


@pytest.fixture(scope="module")
def parquet():
    if not _PARQUET.exists():
        if not _SFT_JSONL.exists():
            pytest.skip("build/sft.jsonl not built")
        subprocess.run([sys.executable, str(_REPO / "echo_verl" / "generate_sft_parquet.py"),
                        "--limit", "8", "--out", str(_PARQUET)], check=True, cwd=_REPO)
    return str(_PARQUET)


@pytest.fixture(scope="module")
def deps():
    from omegaconf import OmegaConf
    from verl.utils import hf_processor, hf_tokenizer
    cfg = OmegaConf.create({"pad_mode": "no_padding", "max_length": 32768, "truncation": "error"})
    return cfg, hf_tokenizer(_MODEL), hf_processor(_MODEL)


def test_upstream_dataset_still_breaks_on_qwen3_vl(parquet, deps):
    """If this ever PASSES, verl fixed it upstream and echo_verl/sft_dataset.py can go.

    Imported in a subprocess: echo_verl.sft_dataset installs its shim into verl's
    module namespace at import time, which would otherwise mask the failure.
    """
    code = (
        "from omegaconf import OmegaConf\n"
        "from verl.utils import hf_tokenizer, hf_processor\n"
        "from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset\n"
        f"M={_MODEL!r}\n"
        "cfg = OmegaConf.create({'pad_mode':'no_padding','max_length':32768,'truncation':'error'})\n"
        f"ds = MultiTurnSFTDataset({parquet!r}, hf_tokenizer(M), cfg, processor=hf_processor(M))\n"
        "ds[0]\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO)
    assert proc.returncode != 0, "upstream MultiTurnSFTDataset now works -- drop the shim"
    assert "IndexError" in proc.stderr


def test_echo_dataset_builds_qwen3_vl_position_ids(parquet, deps):
    from echo_verl.sft_dataset import EchoMultiTurnSFTDataset
    cfg, tokenizer, processor = deps
    ds = EchoMultiTurnSFTDataset(parquet, tokenizer, cfg, processor=processor)
    row = ds[0]

    seq_len = row["input_ids"].shape[0]
    # 1 text row + 3 mRoPE rows (t, h, w)
    assert row["position_ids"].shape == (4, seq_len)
    assert row["loss_mask"].shape == (seq_len,)
    # loss is on assistant turns only -- some, but not all, of the sequence
    assert 0 < int(row["loss_mask"].sum()) < seq_len
    mm = row["multi_modal_inputs"]
    assert "video_grid_thw" in mm and "image_grid_thw" in mm


def test_rope_matches_the_rl_path(parquet, deps):
    """SFT and rollout must construct identical position ids for the same tokens."""
    from echo_verl.sft_dataset import EchoMultiTurnSFTDataset
    cfg, tokenizer, processor = deps
    ds = EchoMultiTurnSFTDataset(parquet, tokenizer, cfg, processor=processor)
    row = ds[0]
    mm = row["multi_modal_inputs"]

    import torch
    # pad_mode=no_padding drops attention_mask from the row; every token is real.
    attention_mask = row.get("attention_mask", torch.ones_like(row["input_ids"]))
    rl_vision, _ = processor.get_rope_index(
        input_ids=row["input_ids"].unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        image_grid_thw=mm.get("image_grid_thw"),
        video_grid_thw=mm.get("video_grid_thw"),
    )
    assert torch.equal(row["position_ids"][1:], rl_vision.squeeze(1))
