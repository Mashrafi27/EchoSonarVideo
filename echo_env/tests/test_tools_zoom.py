from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import zoom


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, min_crop_side=28,
                    n_highres_frames=8, highres_max_side=336)
    return cfg, build_manifest(preprocessed_dir, study_uuid), PILFrameLoader()


def test_zoom_single_frame(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[3])
    assert obs.ok
    assert obs.n_frames == 1
    f = obs.frames[0]
    assert f.kind == "crop"
    assert f.frame_index == 3
    assert f.bbox == (40, 40, 200, 240)
    assert loader.size(f.image) == (160, 200)


def test_zoom_multi_frame(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[1, 3, 5])
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [1, 3, 5]


def test_zoom_defaults_to_midframe(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [40, 40, 200, 240], frame_indices=[])
    assert obs.ok
    assert obs.n_frames == 1  # midframe of the 10-frame clip


def test_zoom_invalid_bbox(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [200, 200, 50, 50], frame_indices=[0])
    assert not obs.ok
    assert "invalid bbox" in obs.error


def test_zoom_unknown_view(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A2C", [0, 0, 50, 50], frame_indices=[0])
    assert not obs.ok
    assert "unknown view" in obs.error


def test_zoom_tiny_bbox_expanded(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = zoom(m, loader, cfg, "A4C", [160, 160, 168, 168], frame_indices=[0])
    assert obs.ok
    w, h = loader.size(obs.frames[0].image)
    assert min(w, h) >= 28
