import json

import pytest

from echo_env.config import EnvConfig
from echo_env.env import EchoEnv


def _env(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_preview_frames=3,
                    max_tool_calls=2, max_total_frames=8, max_calls_per_turn=3)
    env = EchoEnv(cfg)
    return env, study_uuid


def test_reset_overview_thumbnails(study_fixture):
    env, study_uuid = _env(study_fixture)
    obs = env.reset(study_uuid)
    assert obs.ok
    # one thumbnail per fixture view (4 views)
    assert obs.n_frames == 4
    assert all(f.kind == "thumbnail" for f in obs.frames)


def test_step_select_view(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    obs, reward, done, info = env.step(action)
    assert obs.ok
    assert obs.n_frames == 3
    assert reward == 0.0
    assert done is False
    assert info["tool_calls"] == 1


def test_step_answer_terminates(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step("<answer>Normal LV function</answer>")
    assert done is True
    assert info["answer"] == "Normal LV function"


def test_step_zoom(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = ('<tool_call>{"name": "zoom", "arguments": '
              '{"view_name": "A4C", "bbox": [40,40,200,240], "frame_indices": [3]}}</tool_call>')
    obs, reward, done, info = env.step(action)
    assert obs.ok
    assert obs.frames[0].kind == "crop"


def test_budget_exhaustion_blocks_calls(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    a = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    b = '<tool_call>{"name": "select_view", "arguments": {"view_name": "PLAX"}}</tool_call>'
    c = '<tool_call>{"name": "select_view", "arguments": {"view_name": "PSAX Apex"}}</tool_call>'
    env.step(a)
    env.step(b)              # now tool_calls == 2 == max
    obs, reward, done, info = env.step(c)
    assert not obs.ok        # blocked, asks to answer
    assert info["tool_calls"] == 2


def test_step_no_action(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step("just some prose, no tags")
    assert not obs.ok
    assert done is False


def test_step_unknown_tool(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    obs, reward, done, info = env.step(
        '<tool_call>{"name": "teleport", "arguments": {}}</tool_call>')
    assert not obs.ok


def test_step_never_raises_on_scalar_indices(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = ('<tool_call>{"name": "select_frames", "arguments": '
              '{"view_name": "A4C", "indices": 5}}</tool_call>')
    obs, reward, done, info = env.step(action)
    assert not obs.ok


def test_step_never_raises_on_scalar_frame_indices_zoom(study_fixture):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    action = ('<tool_call>{"name": "zoom", "arguments": '
              '{"view_name": "A4C", "bbox": [40,40,200,240], "frame_indices": 5}}</tool_call>')
    obs, reward, done, info = env.step(action)
    # never raises: a scalar frame_indices is treated as empty by the defense-in-depth
    # guard in _clean_indices, and zoom's intentional midframe fallback then applies
    # (same as an explicit frame_indices=[] -- see test_zoom_defaults_to_midframe).
    assert obs.ok
    assert obs.frames[0].kind == "crop"


@pytest.mark.parametrize("tool_name,extra_args", [
    ("select_view", {}),
    ("select_frames", {"indices": [0]}),
    ("zoom", {"bbox": [40, 40, 200, 240], "frame_indices": [0]}),
])
def test_step_never_raises_on_non_str_view_name(study_fixture, tool_name, extra_args):
    env, study_uuid = _env(study_fixture)
    env.reset(study_uuid)
    args = {"view_name": 5, **extra_args}
    action = f'<tool_call>{{"name": "{tool_name}", "arguments": {json.dumps(args)}}}</tool_call>'
    obs, reward, done, info = env.step(action)
    assert not obs.ok


def test_step_truncates_frames_to_remaining_budget(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_preview_frames=3,
                    max_tool_calls=3, max_total_frames=2, max_calls_per_turn=3)
    env = EchoEnv(cfg)
    env.reset(study_uuid)
    action = '<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>'
    obs, reward, done, info = env.step(action)
    # select_view would normally return 3 preview frames, but budget caps total at 2
    assert obs.ok
    assert obs.n_frames == 2
    assert info["total_frames"] == 2

    # a follow-up call has zero frames left in the budget; should not raise,
    # and should not merge in any frames (even though the underlying tool call succeeds)
    action2 = '<tool_call>{"name": "select_view", "arguments": {"view_name": "PLAX"}}</tool_call>'
    obs2, reward2, done2, info2 = env.step(action2)
    assert not obs2.ok


def test_step_before_reset_raises_runtime_error():
    cfg = EnvConfig(preprocessed_dir="/nonexistent")
    env = EchoEnv(cfg)
    with pytest.raises(RuntimeError):
        env.step('<tool_call>{"name": "select_view", "arguments": {"view_name": "A4C"}}</tool_call>')
