import os
import pytest
from PIL import Image


def make_png(path: str, w: int = 336, h: int = 336, marker: int = 0) -> None:
    """Write a solid-color PNG whose R channel encodes `marker` (frame index),
    so tests can read back which frame was returned."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (w, h), (marker % 256, 0, 0))
    img.save(path)


# (clip_dirname, n_frames)
_FIXTURE_CLIPS = [
    ("di-AAAA-0001_A4C", 10),
    ("di-BBBB-0002_A4C Zoomed Mitral", 6),
    ("di-CCCC-0003_PLAX", 8),
    ("di-DDDD-0004_PSAX Apex", 4),
]


@pytest.fixture
def study_fixture(tmp_path):
    """Create a real on-disk study and return (preprocessed_dir, study_uuid)."""
    preprocessed_dir = str(tmp_path / "preprocessed_data")
    study_uuid = "st-TEST-0000-0000"
    study_dir = os.path.join(preprocessed_dir, study_uuid)
    for clip_name, n in _FIXTURE_CLIPS:
        for i in range(n):
            make_png(os.path.join(study_dir, clip_name, f"{i}.png"), marker=i)
    return preprocessed_dir, study_uuid
