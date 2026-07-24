from echo_env.config import EnvConfig
from echo_env.frames import PILFrameLoader
from echo_env.manifest import build_manifest
from echo_env.tools import select_frames


def _setup(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    cfg = EnvConfig(preprocessed_dir=preprocessed_dir, n_highres_frames=8, highres_max_side=336)
    return cfg, build_manifest(preprocessed_dir, study_uuid), PILFrameLoader()


def test_select_specific_frames(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [2, 5, 7])
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [2, 5, 7]
    assert all(f.kind == "highres" for f in obs.frames)
    # native resolution preserved
    assert loader.size(obs.frames[0].image) == (336, 336)
    # marker check: frame 2's R channel == 2
    assert obs.frames[0].image.getpixel((0, 0))[0] == 2


def test_dedup_sort_and_drop_out_of_range(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [5, 5, 2, 99])  # 99 out of 0..9
    assert obs.ok
    assert [f.frame_index for f in obs.frames] == [2, 5]


def test_cap_at_n_highres(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    cfg.n_highres_frames = 3
    obs = select_frames(m, loader, cfg, "A4C", [0, 1, 2, 3, 4, 5])
    assert obs.n_frames == 3
    assert [f.frame_index for f in obs.frames] == [0, 1, 2]


def test_all_invalid_indices(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A4C", [99, -1, 200])
    assert not obs.ok


def test_unknown_view(study_fixture):
    cfg, m, loader = _setup(study_fixture)
    obs = select_frames(m, loader, cfg, "A2C", [0])
    assert not obs.ok
    assert "unknown view" in obs.error
