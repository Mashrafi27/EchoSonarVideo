import h5py
import numpy as np
import pytest

from echoprime_track.darya_cache import (
    detr_class_ids_present,
    load_clip_tokens,
    load_detr_tokens,
)


@pytest.fixture
def clip_h5_path(tmp_path):
    """Matches the real on-disk convention (confirmed against report_generation/data/
    clip_tokens_train.h5 directly): clip_h5[dicom_uuid] is a GROUP with a "tokens" dataset,
    not a bare dataset at the top level."""
    path = tmp_path / "clip.h5"
    with h5py.File(path, "w") as f:
        f.create_group("di-present").create_dataset(
            "tokens", data=np.arange(393 * 768, dtype=np.float32).reshape(393, 768))
    return str(path)


@pytest.fixture
def detr_h5_path(tmp_path):
    """Matches the real on-disk convention (confirmed against report_generation/data/
    train_detections.h5 directly): detr_h5[dicom_uuid]["frame_N"] has "classes" (k,) and
    "embeddings" (k, 256) datasets."""
    path = tmp_path / "detr.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("di-with-detections")
        frame0 = grp.create_group("frame_0")
        frame0.create_dataset("classes", data=np.array([1, 4], dtype=np.int64))
        frame0.create_dataset(
            "embeddings",
            data=np.stack([np.full(256, 1.0, dtype=np.float32),
                            np.full(256, 4.0, dtype=np.float32)]))
        frame1 = grp.create_group("frame_1")
        frame1.create_dataset("classes", data=np.array([1], dtype=np.int64))
        frame1.create_dataset(
            "embeddings", data=np.full((1, 256), 3.0, dtype=np.float32))
        # dicom with no detections at all -- an empty group, no frame_* children.
        f.create_group("di-no-detections")
    return str(path)


def test_load_clip_tokens_returns_none_for_missing_dicom(clip_h5_path):
    with h5py.File(clip_h5_path, "r") as clip_h5:
        assert load_clip_tokens(clip_h5, "di-absent") is None


def test_load_clip_tokens_returns_real_array_for_present_dicom(clip_h5_path):
    with h5py.File(clip_h5_path, "r") as clip_h5:
        tokens = load_clip_tokens(clip_h5, "di-present")
    assert tokens.shape == (393, 768)
    assert tokens[0, 0] == 0.0
    assert tokens[1, 0] == 768.0


def test_detr_class_ids_present_empty_for_no_detections(detr_h5_path):
    with h5py.File(detr_h5_path, "r") as detr_h5:
        assert detr_class_ids_present(detr_h5, "di-no-detections") == []


def test_detr_class_ids_present_empty_for_missing_dicom(detr_h5_path):
    with h5py.File(detr_h5_path, "r") as detr_h5:
        assert detr_class_ids_present(detr_h5, "di-absent") == []


def test_detr_class_ids_present_returns_sorted_class_list(detr_h5_path):
    with h5py.File(detr_h5_path, "r") as detr_h5:
        assert detr_class_ids_present(detr_h5, "di-with-detections") == [1, 4]


def test_load_detr_tokens_empty_array_for_no_detections(detr_h5_path):
    with h5py.File(detr_h5_path, "r") as detr_h5:
        tokens = load_detr_tokens(detr_h5, "di-no-detections")
    assert tokens.shape == (0, 256)


def test_load_detr_tokens_mean_pools_multiple_frames_per_class(detr_h5_path):
    """class 1 appears in frame_0 (value 1.0) and frame_1 (value 3.0) -> mean 2.0.
    class 4 appears only in frame_0 (value 4.0) -> mean 4.0. Rows ordered by
    class_ids_present ([1, 4])."""
    with h5py.File(detr_h5_path, "r") as detr_h5:
        tokens = load_detr_tokens(detr_h5, "di-with-detections")
    assert tokens.shape == (2, 256)
    assert np.allclose(tokens[0], 2.0)
    assert np.allclose(tokens[1], 4.0)
