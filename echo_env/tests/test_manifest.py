from echo_env.manifest import build_manifest, StudyManifest, ViewEntry


def test_build_manifest_lists_views(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    names = m.view_names()
    assert "A4C" in names
    assert "A4C Zoomed Mitral" in names
    assert "PLAX" in names


def test_resolve_exact_case_insensitive(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("a4c").view_name == "A4C"
    assert m.resolve("A4C Zoomed Mitral").view_name == "A4C Zoomed Mitral"


def test_resolve_base_fallback_prefers_plain(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    # "A4C" plain clip exists, so a base query returns it, not the zoomed variant
    assert m.resolve("A4C").view_name == "A4C"


def test_resolve_unknown_returns_none(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("A2C") is None


def test_frame_count(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    assert m.resolve("A4C").frame_count == 10


def test_overview_deterministic_limit(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    m = build_manifest(preprocessed_dir, study_uuid)
    ov = m.overview(limit=2)
    assert len(ov) == 2
    assert [v.view_name for v in ov] == sorted(m.view_names())[:2]
