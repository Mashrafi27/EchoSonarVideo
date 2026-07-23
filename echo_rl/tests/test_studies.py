import os
import pytest
from echo_rl.data.studies import index_study, study_dir


def _make_study(tmp_path):
    root = tmp_path / "st-TEST-0000-0000"
    clip = root / "di-AAAA-BBBB-CCCC_A4C"
    clip.mkdir(parents=True)
    # deliberately out of lexical order to prove numeric sort
    for i in [0, 1, 2, 10, 11]:
        (clip / f"{i}.png").write_bytes(b"x")
    (root / "di-DDDD-EEEE-FFFF_PLAX Standard").mkdir()  # empty clip -> skipped
    for i in [0, 1]:
        (root / "di-DDDD-EEEE-FFFF_PLAX Standard" / f"{i}.png").write_bytes(b"x")
    return str(root)


def test_index_study(tmp_path):
    root = _make_study(tmp_path)
    clips = index_study(root)
    views = sorted(c.view for c in clips)
    assert views == ["A4C", "PLAX Standard"]
    a4c = next(c for c in clips if c.view == "A4C")
    assert a4c.frame_count == 5
    assert a4c.frame_files == ["0.png", "1.png", "2.png", "10.png", "11.png"]
    assert a4c.frame_path(3).endswith("/10.png")


def test_index_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_study(str(tmp_path / "nope"))


def test_study_dir():
    assert study_dir("/data", "st-1").endswith("/data/st-1")
