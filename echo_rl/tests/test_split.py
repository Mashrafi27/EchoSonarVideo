import json
from echo_rl.data.split import assign_split, test_study_set


def test_assign_split_explicit():
    assert assign_split("st-x", "TEST", set()) == "test"
    assert assign_split("st-x", "VAL", set()) == "val"
    assert assign_split("st-x", "TRAIN", set()) == "train"


def test_assign_split_test_set_wins():
    assert assign_split("st-x", "TRAIN", {"st-x"}) == "test"


def test_assign_split_hash_deterministic():
    a = assign_split("st-abc", None, set())
    b = assign_split("st-abc", None, set())
    assert a == b and a in {"train", "val"}


def test_test_study_set(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"study_uuid": "st-1"}) + "\n" + json.dumps({"study_uuid": "st-1"}) + "\n"
                 + json.dumps({"study_uuid": "st-2"}) + "\n")
    assert test_study_set(str(p)) == {"st-1", "st-2"}
