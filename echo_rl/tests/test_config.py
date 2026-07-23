import os
from echo_rl.config import Config


def test_from_env_defaults():
    cfg = Config.from_env()
    assert cfg.preprocessed_dir.endswith("preprocessed_data")
    assert cfg.vqa_train.endswith("train_vqa_with_thinking.jsonl")
    assert cfg.n_preview_frames == 5
    assert cfg.n_highres_frames == 8
    assert cfg.study_split.endswith("echojepa_study_split_full.csv")


def test_from_env_override(monkeypatch):
    monkeypatch.setenv("ECHO_PREPROCESSED_DIR", "/tmp/foo")
    cfg = Config.from_env()
    assert cfg.preprocessed_dir == "/tmp/foo"


def test_study_split_env_override(monkeypatch):
    monkeypatch.setenv("ECHO_STUDY_SPLIT", "/tmp/custom_split.csv")
    cfg = Config.from_env()
    assert cfg.study_split == "/tmp/custom_split.csv"
