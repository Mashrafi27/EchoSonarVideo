import json

import h5py
import numpy as np
import pytest

from echoprime_track.generate_grpo_parquet import (
    VERIFIABLE_QUESTION_TYPES,
    build_dicoms_by_view,
    build_row,
    write_parquet,
)


@pytest.fixture
def small_h5s(tmp_path):
    clip_path = tmp_path / "clip.h5"
    detr_path = tmp_path / "detr.h5"
    with h5py.File(clip_path, "w") as f:
        # NOTE: fixed from the brief's literal `f.create_dataset("dicom-A4C", data=...)` --
        # confirmed against the real on-disk file (report_generation/data/clip_tokens_train.h5)
        # and against darya_cache.load_clip_tokens's own `clip_h5[dicom_uuid]["tokens"][:]`
        # lookup that clip_h5[dicom_uuid] must be a GROUP containing a "tokens" dataset, not a
        # bare dataset at the top level -- the brief's original fixture line raised
        # `ValueError: Field names only allowed for compound types` against the real
        # load_clip_tokens implementation given in the same brief.
        f.create_group("dicom-A4C").create_dataset(
            "tokens", data=np.zeros((393, 768), dtype=np.float32))
    with h5py.File(detr_path, "w") as f:
        pass  # no detections for this dicom -- must be handled, not required
    return str(clip_path), str(detr_path)


def _rl_rec(question_type="abnormality_classification"):
    return {
        "study_uuid": "study-1",
        "question": "Is there mitral regurgitation?",
        "question_type": question_type,
        "reward_key": {"kind": "yesno", "target": "no", "gold": {}},
        "dicoms_by_view": {"A4C": "dicom-A4C"},
    }


def test_build_row_includes_verifiable_question_types(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec("abnormality_classification"), clip_h5, detr_h5)
    assert row is not None
    assert row["data_source"] == "echoprime_grpo"
    assert "A4C:" in row["prompt"][1]["content"]
    assert row["extra_info"]["dicom_uuids_by_view"] == {"A4C": "dicom-A4C"}


def test_build_row_excludes_unverifiable_question_types(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec("structure_description"), clip_h5, detr_h5)
    assert row is None


def test_verifiable_question_types_is_exactly_two():
    assert VERIFIABLE_QUESTION_TYPES == {"abnormality_classification", "abnormality_list"}


def test_build_row_prompt_has_clip_token_count_matching_grid(small_h5s):
    clip_path, detr_path = small_h5s
    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec(), clip_h5, detr_h5)
    from echoprime_track.modeling import CLIP_TOKEN
    assert row["prompt"][1]["content"].count(CLIP_TOKEN) == 393


def test_build_row_includes_structure_features_when_detections_present(tmp_path):
    clip_path = tmp_path / "clip2.h5"
    detr_path = tmp_path / "detr2.h5"
    with h5py.File(clip_path, "w") as f:
        f.create_group("dicom-A4C").create_dataset(
            "tokens", data=np.zeros((393, 768), dtype=np.float32))
    with h5py.File(detr_path, "w") as f:
        grp = f.create_group("dicom-A4C")
        frame0 = grp.create_group("frame_0")
        frame0.create_dataset("classes", data=np.array([1], dtype=np.int64))
        frame0.create_dataset("embeddings", data=np.zeros((1, 256), dtype=np.float32))

    with h5py.File(clip_path, "r") as clip_h5, h5py.File(detr_path, "r") as detr_h5:
        row = build_row(_rl_rec(), clip_h5, detr_h5)

    from echoprime_track.modeling import DETR_TOKEN
    content = row["prompt"][1]["content"]
    assert "Structure features:" in content
    assert content.count(DETR_TOKEN) == 1


# -- dicoms_by_view grouping (Step 3's post-Step-3 note: build/rl.jsonl carries
# overview.views[], not dicoms_by_view directly -- main() must build it) -------------------

def test_build_dicoms_by_view_prefers_clip_covered_candidate(tmp_path):
    clip_path = tmp_path / "clip3.h5"
    with h5py.File(clip_path, "w") as f:
        f.create_group("di-covered").create_dataset(
            "tokens", data=np.zeros((393, 768), dtype=np.float32))

    views = [
        {"view": "A4C", "frame": "/x/study1/di-uncovered_A4C/14.png", "frame_count": 29},
        {"view": "A4C", "frame": "/x/study1/di-covered_A4C/14.png", "frame_count": 28},
        {"view": "PLAX Standard", "frame": "/x/study1/di-only_PLAX Standard/13.png",
         "frame_count": 27},
    ]
    with h5py.File(clip_path, "r") as clip_h5:
        result = build_dicoms_by_view(views, clip_h5)

    assert result == {"A4C": "di-covered", "PLAX Standard": "di-only"}


def test_build_dicoms_by_view_falls_back_to_first_when_none_covered(tmp_path):
    clip_path = tmp_path / "clip4.h5"
    with h5py.File(clip_path, "w") as f:
        pass  # empty -- nothing covered

    views = [
        {"view": "A4C", "frame": "/x/study1/di-first_A4C/14.png", "frame_count": 29},
        {"view": "A4C", "frame": "/x/study1/di-second_A4C/14.png", "frame_count": 28},
    ]
    with h5py.File(clip_path, "r") as clip_h5:
        result = build_dicoms_by_view(views, clip_h5)

    assert result == {"A4C": "di-first"}


def test_build_dicoms_by_view_preserves_first_seen_view_order(tmp_path):
    clip_path = tmp_path / "clip5.h5"
    with h5py.File(clip_path, "w") as f:
        pass

    views = [
        {"view": "PSAX Apex", "frame": "/x/s/di-c_PSAX Apex/1.png", "frame_count": 1},
        {"view": "A4C", "frame": "/x/s/di-a_A4C/1.png", "frame_count": 1},
        {"view": "PLAX Standard", "frame": "/x/s/di-b_PLAX Standard/1.png", "frame_count": 1},
    ]
    with h5py.File(clip_path, "r") as clip_h5:
        result = build_dicoms_by_view(views, clip_h5)

    assert list(result.keys()) == ["PSAX Apex", "A4C", "PLAX Standard"]


# -- write_parquet: dicom_uuids_by_view order must survive the parquet round-trip -----------
#
# Found this session: `pa.Table.from_pylist` infers a STRUCT type for a nested dict field by
# unioning ALL keys seen across every row (alphabetically), nulling missing ones per row --
# it does NOT preserve each row's own insertion order. Since different studies have different
# view sets, a raw dict here would desync the per-view h5-read order (rl_dataset.py) from the
# CLIP_TOKEN/DETR_TOKEN placement order baked into the prompt TEXT at generation time. write_
# parquet must JSON-encode this field (matching reward_model.ground_truth's existing
# convention in this same module) instead of leaving it a raw dict.

def test_write_parquet_preserves_dicom_uuids_by_view_order(tmp_path):
    rows = [
        {
            "data_source": "echoprime_grpo", "agent_name": "echoprime_tool_agent",
            "prompt": [{"role": "system", "content": "sys"},
                       {"role": "user", "content": "u1"}],
            "reward_model": {"ground_truth": "{}", "style": "rule"},
            "ability": "echo_vqa",
            "extra_info": {"study_uuid": "s1", "question_type": "abnormality_list",
                            "dicom_uuids_by_view": {"PSAX Apex": "d-c", "A4C": "d-a",
                                                     "PLAX Standard": "d-b"},
                            "need_tools_kwargs": False},
        },
        {
            "data_source": "echoprime_grpo", "agent_name": "echoprime_tool_agent",
            "prompt": [{"role": "system", "content": "sys"},
                       {"role": "user", "content": "u2"}],
            "reward_model": {"ground_truth": "{}", "style": "rule"},
            "ability": "echo_vqa",
            "extra_info": {"study_uuid": "s2", "question_type": "abnormality_classification",
                            "dicom_uuids_by_view": {"Subcostal Standard": "d-x", "A2C": "d-y"},
                            "need_tools_kwargs": False},
        },
    ]
    out = tmp_path / "out.parquet"
    n = write_parquet(rows, str(out))
    assert n == 2

    import pyarrow.parquet as pq
    read_back = pq.read_table(str(out)).to_pylist()
    d0 = json.loads(read_back[0]["extra_info"]["dicom_uuids_by_view"])
    d1 = json.loads(read_back[1]["extra_info"]["dicom_uuids_by_view"])
    assert list(d0.items()) == [("PSAX Apex", "d-c"), ("A4C", "d-a"), ("PLAX Standard", "d-b")]
    assert list(d1.items()) == [("Subcostal Standard", "d-x"), ("A2C", "d-y")]
