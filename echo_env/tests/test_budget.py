from echo_env.config import EnvConfig
from echo_env.budget import Budget
from echo_env.observation import Observation, FrameImg


def _obs(n):
    return Observation(tool="select_view",
                       frames=[FrameImg(object(), "A4C", i, "preview") for i in range(n)])


def test_can_call_limit():
    b = Budget(EnvConfig(max_tool_calls=2))
    assert b.can_call()
    b.register("select_view", {"view_name": "A4C"}, _obs(3))
    b.register("select_view", {"view_name": "PLAX"}, _obs(3))
    assert not b.can_call()


def test_frames_left():
    b = Budget(EnvConfig(max_total_frames=10))
    b.register("select_view", {"view_name": "A4C"}, _obs(4))
    assert b.frames_left() == 6


def test_dedup_signature():
    b = Budget(EnvConfig())
    assert not b.seen_before("select_view", {"view_name": "A4C"})
    b.register("select_view", {"view_name": "A4C"}, _obs(1))
    assert b.seen_before("select_view", {"view_name": "A4C"})
    assert not b.seen_before("select_view", {"view_name": "PLAX"})
