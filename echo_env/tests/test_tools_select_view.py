from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import select_view


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_preview_frames=5, preview_max_side=160)
    m = build_manifest(preprocessed_dir, study_uuid)
    return cfg, m, PILFrameLoader()


def test_select_view_returns_preview_frames(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "A4C")   # 10-frame clip
    assert obs.ok
    assert obs.n_frames == 5
    assert all(f.kind == "preview" for f in obs.frames)
    # downscaled to preview_max_side
    assert loader.size(obs.frames[0].image) == (160, 160)
    # evenly spaced across 10 frames -> deterministic indices
    idxs = [f.frame_index for f in obs.frames]
    assert idxs == sorted(idxs) and idxs[0] == 0 and idxs[-1] == 9


def test_select_view_short_clip(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "PSAX Apex")  # 4 frames < 5 requested
    assert obs.ok
    assert obs.n_frames <= 4


def test_select_view_unknown(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_view(m, loader, cfg, "A2C")
    assert not obs.ok
    assert "unknown view" in obs.error
