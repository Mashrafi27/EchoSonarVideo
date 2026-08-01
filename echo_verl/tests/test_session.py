import pytest
from echo_env.config import EnvConfig
from echo_verl.session import EchoSession


def _cfg(preprocessed_dir):
    return EnvConfig(preprocessed_dir=preprocessed_dir, min_crop_side=32,
                     highres_max_side=320, preview_max_side=160)


def test_select_view_returns_frames(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_view", {"view_name": "A4C"})
    assert obs.ok
    assert obs.n_frames > 0


def test_select_frames_op(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_frames", {"view_name": "A4C", "indices": [0, 2]})
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [0, 2]


def test_zoom_op(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("zoom", {"view_name": "A4C", "bbox": [40, 40, 200, 240], "frame_indices": [1]})
    assert obs.ok
    assert obs.frames[0].kind == "crop"


def test_unknown_op_fails_cleanly(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("teleport", {"view_name": "A4C"})
    assert not obs.ok
    assert "unknown op" in obs.error


def test_missing_view_name_does_not_raise(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("select_view", {})           # no view_name
    assert not obs.ok                         # failure Observation, never an exception


def test_zoom_missing_bbox_does_not_raise(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    s = EchoSession(_cfg(preprocessed_dir), study_uuid)
    obs = s.run("zoom", {"view_name": "A4C", "frame_indices": [0]})  # no bbox
    assert not obs.ok            # failure Observation, never an exception
