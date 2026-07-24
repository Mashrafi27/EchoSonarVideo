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
