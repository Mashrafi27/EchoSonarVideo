import os
from echo_env.frames import PILFrameLoader


def test_load_and_size(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "3.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    assert loader.size(img) == (320, 320)
    # marker: R channel encodes the frame index (3)
    assert img.getpixel((0, 0))[0] == 3


def test_downscale_shrinks_large_only(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "0.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    small = loader.downscale(img, 168)
    assert loader.size(small) == (168, 168)
    # already-small stays put
    same = loader.downscale(small, 336)
    assert loader.size(same) == (168, 168)


def test_crop(study_fixture):
    preprocessed_dir, study_uuid = study_fixture
    p = os.path.join(preprocessed_dir, study_uuid, "di-AAAA-0001_A4C", "0.png")
    loader = PILFrameLoader()
    img = loader.load(p)
    c = loader.crop(img, (10, 10, 100, 120))
    assert loader.size(c) == (90, 110)
