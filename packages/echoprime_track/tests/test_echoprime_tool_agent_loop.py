import torch

from echoprime_track.echoprime_tool_agent_loop import resolve_tool_call


def _fake_grid():
    # (393, 768) grid where every token's first value is its own index, for easy assertion
    g = torch.zeros(393, 768)
    for i in range(393):
        g[i, 0] = i
    return g


def test_resolve_select_frames_returns_group_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A4C", "frame_indices": [0, 1]}},
        view_grids={"A4C": grid})
    assert tokens.shape == (49, 768)
    assert tokens[0, 0].item() == 1  # group 0 starts at absolute index 1
    assert "A4C" in text


def test_resolve_zoom_returns_spatial_subset():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "zoom", "arguments": {"view": "A4C", "bbox": [0.0, 0.0, 0.5, 0.5],
                                        "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert 0 < tokens.shape[0] < 49
    assert tokens.shape[1] == 768


def test_resolve_unknown_view_returns_error_text_no_tokens():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_frames", "arguments": {"view": "A2C", "frame_indices": [0]}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown view" in text


def test_resolve_unknown_tool_name_returns_error():
    grid = _fake_grid()
    tokens, text = resolve_tool_call(
        {"name": "select_view", "arguments": {"view": "A4C"}},
        view_grids={"A4C": grid})
    assert tokens is None
    assert "unknown tool" in text
