import json
from echo_rl.data.split import assign_split, load_test_studies, load_canonical_split


def test_load_canonical_split(tmp_path):
    p = tmp_path / "split.csv"
    p.write_text("study_uuid,split\nst-x,TRAIN\nst-y,val\nst-z,Test\n")
    assert load_canonical_split(str(p)) == {"st-x": "TRAIN", "st-y": "VAL", "st-z": "TEST"}


def test_load_canonical_split_missing_file(tmp_path):
    assert load_canonical_split(str(tmp_path / "nope.csv")) == {}


def test_assign_split_from_canonical():
    assert assign_split("st-x", {"st-x": "TEST"}) == "test"
    assert assign_split("st-x", {"st-x": "VAL"}) == "val"
    assert assign_split("st-x", {"st-x": "TRAIN"}) == "train"


def test_assign_split_hash_fallback_deterministic():
    a = assign_split("st-abc", {})
    b = assign_split("st-abc", {})
    assert a == b and a in {"train", "val"}


def test_load_test_studies(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"study_uuid": "st-1"}) + "\n" + json.dumps({"study_uuid": "st-1"}) + "\n"
                 + json.dumps({"study_uuid": "st-2"}) + "\n")
    assert load_test_studies(str(p)) == {"st-1", "st-2"}
