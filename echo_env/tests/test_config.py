import os
from echo_env.config import EnvConfig


def test_defaults():
    c = EnvConfig()
    assert c.n_preview_frames == 5
    assert c.highres_max_side == 320   # Qwen3-VL: 10 x 32px merged patch
    assert c.min_crop_side == 32
    assert c.max_tool_calls == 8
    assert c.max_total_frames == 32
    assert c.seed == 0


def test_from_env_reads_preprocessed_dir(monkeypatch):
    monkeypatch.setenv("ECHO_PREPROCESSED_DIR", "/some/where")
    monkeypatch.setenv("ECHO_MAX_TOOL_CALLS", "3")
    c = EnvConfig.from_env()
    assert c.preprocessed_dir == "/some/where"
    assert c.max_tool_calls == 3


def test_from_env_default_preprocessed_dir_endswith(monkeypatch):
    monkeypatch.delenv("ECHO_PREPROCESSED_DIR", raising=False)
    c = EnvConfig.from_env()
    assert c.preprocessed_dir.endswith("preprocessed_data")
