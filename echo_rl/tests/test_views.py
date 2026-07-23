import pytest
from echo_rl.data.views import parse_clip_dirname, canonical_view, base_view


def test_parse_clip_dirname():
    assert parse_clip_dirname("di-2040-A343-7E7F_PLAX Standard") == ("di-2040-A343-7E7F", "PLAX Standard")
    assert parse_clip_dirname("di-4503-8690-7008_PSAXA Zoomed Aorta")[1] == "PSAXA Zoomed Aorta"


def test_parse_clip_dirname_no_underscore():
    with pytest.raises(ValueError):
        parse_clip_dirname("di-2040-A343-7E7F")


def test_canonical_view():
    assert canonical_view("  A4C   Zoomed Mitral ") == "A4C Zoomed Mitral"


def test_base_view():
    assert base_view("PLAX Standard") == "PLAX"
    assert base_view("A4C") == "A4C"
