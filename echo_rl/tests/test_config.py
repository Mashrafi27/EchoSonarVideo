import os
from echo_rl.config import Config


def test_from_env_defaults():
    cfg = Config.from_env()
    assert cfg.preprocessed_dir.endswith("preprocessed_data")
    assert cfg.vqa_train.endswith("train_vqa_with_thinking.jsonl")
    assert cfg.n_preview_frames == 5
    assert cfg.n_highres_frames == 8


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("ECHO_PREPROCESSED_DIR", "/tmp/foo")
    cfg = Config.from_env()
    assert cfg.preprocessed_dir == "/tmp/foo"
