"""Sampling must be reproducible and must not starve the rare question types."""
import json

from echo_verl.sample import build_manifest, parse_caps, record_id, select


def _corpus(n_studies=5):
    """Mirrors the real shape: many of two types, exactly one of each report type."""
    recs = []
    for s in range(n_studies):
        uuid = f"st-{s:04d}"
        for i in range(11):
            recs.append({"study_uuid": uuid, "question_type": "structure_description",
                         "question": f"sd {i}"})
            recs.append({"study_uuid": uuid, "question_type": "abnormality_classification",
                         "question": f"ac {i}"})
        for qt in ("abnormality_list", "conclusion", "full_report"):
            recs.append({"study_uuid": uuid, "question_type": qt, "question": qt})
    return recs


def test_same_seed_selects_the_same_records():
    recs = _corpus()
    caps = {"structure_description": 3, "abnormality_classification": 3}
    a = select(recs, caps=caps, seed=7)
    b = select(recs, caps=caps, seed=7)
    assert [record_id(r) for r in a] == [record_id(r) for r in b]


def test_selection_is_independent_of_input_order():
    """Reproducibility must not depend on how sft.jsonl happened to be written."""
    recs = _corpus()
    caps = {"structure_description": 3}
    forward = select(recs, caps=caps, seed=7)
    backward = select(list(reversed(recs)), caps=caps, seed=7)
    assert [record_id(r) for r in forward] == [record_id(r) for r in backward]


def test_different_seeds_select_different_records():
    recs = _corpus()
    caps = {"structure_description": 3}
    a = {record_id(r) for r in select(recs, caps=caps, seed=1)}
    b = {record_id(r) for r in select(recs, caps=caps, seed=2)}
    assert a != b


def test_uncapped_types_are_kept_whole():
    """The three report types are one per study; a cap on the bulk types must not
    touch them. This is the failure a uniform per-study cap would cause."""
    recs = _corpus(n_studies=5)
    out = select(recs, caps={"structure_description": 2,
                             "abnormality_classification": 2}, seed=0)
    counts = {}
    for r in out:
        counts[r["question_type"]] = counts.get(r["question_type"], 0) + 1
    assert counts["full_report"] == 5
    assert counts["conclusion"] == 5
    assert counts["abnormality_list"] == 5
    assert counts["structure_description"] == 10
    assert counts["abnormality_classification"] == 10


def test_cap_above_available_keeps_everything():
    recs = _corpus(n_studies=2)
    out = select(recs, caps={"structure_description": 999}, seed=0)
    assert sum(1 for r in out if r["question_type"] == "structure_description") == 22


def test_adding_a_study_does_not_reshuffle_existing_ones():
    """Group-derived seeds mean an unrelated study joining the corpus cannot change
    which records an existing study contributes."""
    caps = {"structure_description": 3}
    small = select(_corpus(n_studies=3), caps=caps, seed=5)
    large = select(_corpus(n_studies=4), caps=caps, seed=5)
    kept = {record_id(r) for r in large}
    assert {record_id(r) for r in small} <= kept


def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    src = tmp_path / "sft.jsonl"
    recs = _corpus(n_studies=3)
    src.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    caps = {"structure_description": 2}
    sel = select(recs, caps=caps, seed=11)
    m = build_manifest(selected=sel, source=src, caps=caps, default_cap=None,
                       seed=11, out_path="o.parquet", ids_path="o.ids.txt",
                       n_source_records=len(recs))
    assert m["params"]["seed"] == 11
    assert m["params"]["per_type_caps"] == caps
    assert m["source"]["sha256"] and m["source"]["n_records"] == len(recs)
    assert m["selected"]["n_records"] == len(sel)
    assert m["selected"]["n_studies"] == 3
    # Same inputs -> same ids hash, which is how a later run proves it matched.
    again = build_manifest(selected=select(recs, caps=caps, seed=11), source=src,
                           caps=caps, default_cap=None, seed=11,
                           out_path="o.parquet", ids_path="o.ids.txt",
                           n_source_records=len(recs))
    assert m["selected"]["ids_sha256"] == again["selected"]["ids_sha256"]


def test_parse_caps():
    assert parse_caps(["a=3", "b=1"]) == {"a": 3, "b": 1}
    try:
        parse_caps(["oops"])
    except ValueError:
        return
    raise AssertionError("malformed --per-type was accepted")
