from echo_env.observation import FrameImg, Observation


def test_frameimg_fields():
    f = FrameImg(image=object(), view_name="A4C", frame_index=3, kind="highres")
    assert f.frame_index == 3
    assert f.bbox is None


def test_observation_counts():
    obs = Observation(tool="select_view", frames=[
        FrameImg(image=object(), view_name="A4C", frame_index=0, kind="preview"),
        FrameImg(image=object(), view_name="A4C", frame_index=5, kind="preview"),
    ], text="ok")
    assert obs.n_frames == 2
    assert obs.ok is True


def test_observation_failure():
    obs = Observation.failure("zoom", "bad bbox")
    assert obs.ok is False
    assert obs.n_frames == 0
    assert obs.error == "bad bbox"
    assert obs.text == "bad bbox"
