from echo_env.observation import Observation, FrameImg
from echo_env.packaging import to_deepeyes_obs


def test_packaging_multi_image():
    obs = Observation(tool="step", frames=[
        FrameImg(image="IMG0", view_name="A4C", frame_index=0, kind="preview"),
        FrameImg(image="IMG1", view_name="A4C", frame_index=5, kind="preview"),
    ], text="ok")
    out = to_deepeyes_obs(obs, user_prompt="Continue.")
    assert out["multi_modal_data"]["image"] == ["IMG0", "IMG1"]
    assert out["prompt"].count("<image>") == 2
    assert out["prompt"].endswith("<|im_start|>assistant\n")
    assert "Continue." in out["prompt"]


def test_packaging_failure_is_text_only():
    obs = Observation.failure("zoom", "invalid bbox [1,2,3,4]")
    out = to_deepeyes_obs(obs, user_prompt="Continue.")
    assert "multi_modal_data" not in out
    assert "Error: invalid bbox" in out["prompt"]
