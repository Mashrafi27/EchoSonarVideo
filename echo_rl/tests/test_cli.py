import json, os
from echo_rl.cli import run


def _clip(pp, study_uuid):
    clip = pp / study_uuid / "di-2_A4C"
    clip.mkdir(parents=True)
    for i in range(6):
        (clip / f"{i}.png").write_bytes(b"x")


def _qa_line(study_uuid):
    return json.dumps({
        "study_uuid": study_uuid, "question_type": "abnormality_classification",
        "thinking": "#### 1. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n",
        "messages": json.dumps([{"role": "user", "content": "Q?"},
                                {"role": "assistant", "content": "No, none."}])})


def _setup(tmp_path, study_uuids, gold_rows, missing_uuid=None, canonical_rows=None):
    """Create fake preprocessed_data + train.jsonl covering `study_uuids` (each with a
    folder/clip) plus one optional record whose study has NO folder (missing_uuid).
    `canonical_rows` is a list of (study_uuid, SPLIT) written to a canonical split CSV;
    defaults to marking every study TRAIN if not given."""
    pp = tmp_path / "pp"
    for uuid in study_uuids:
        _clip(pp, uuid)

    lines = [_qa_line(u) for u in study_uuids]
    if missing_uuid:
        lines.append(_qa_line(missing_uuid))
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(lines) + "\n")

    test = tmp_path / "test.jsonl"
    test.write_text("")

    gold = tmp_path / "gold"
    gold.mkdir()
    header = "study_id,label,study designation,from_measurement,from_text,text,text_fields\n"
    rows = "".join(f"{sid},0,{desig},T,T,t,c\n" for sid, desig in gold_rows)
    (gold / "heart_failure_classification.csv").write_text(header + rows)

    if canonical_rows is None:
        canonical_rows = [(u, "TRAIN") for u in study_uuids]
    split_csv = tmp_path / "study_split.csv"
    split_lines = ["study_uuid,split"] + [f"{u},{sp}" for u, sp in canonical_rows]
    split_csv.write_text("\n".join(split_lines) + "\n")

    out = tmp_path / "out"
    env = {"ECHO_PREPROCESSED_DIR": str(pp), "ECHO_VQA_TRAIN": str(train),
           "ECHO_VQA_TEST": str(test), "ECHO_GOLD_DIR": str(gold), "ECHO_OUT_DIR": str(out),
           "ECHO_STUDY_SPLIT": str(split_csv)}
    return env, out


def _apply_env(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_build_sft_and_rl(tmp_path, monkeypatch):
    env, out = _setup(tmp_path, ["st-1"], [("st-1", "TRAIN")], canonical_rows=[("st-1", "TRAIN")])
    _apply_env(monkeypatch, env)
    run(["build-sft"])
    run(["build-rl"])
    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 1 and sft[0]["trajectory"]["turns"][0]["args"]["view"] == "A4C"
    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert rl[0]["reward_key"]["kind"] == "yesno"
    pool = json.load(open(out / "rl_pool.json"))
    assert isinstance(pool["indices"], list) and len(pool["indices"]) > 0


def test_missing_folder_is_skipped(tmp_path, monkeypatch):
    # st-1 and st-2 have folders; st-missing has a QA record but no preprocessed folder.
    env, out = _setup(
        tmp_path, ["st-1", "st-2"], [("st-1", "TRAIN"), ("st-2", "TRAIN")],
        missing_uuid="st-missing",
        canonical_rows=[("st-1", "TRAIN"), ("st-2", "TRAIN"), ("st-missing", "TRAIN")])
    _apply_env(monkeypatch, env)
    run(["build-sft"])
    run(["build-rl"])

    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 2
    assert {r["study_uuid"] for r in sft} == {"st-1", "st-2"}

    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert len(rl) == 2
    assert {r["study_uuid"] for r in rl} == {"st-1", "st-2"}


def test_build_sft_limit_truncates(tmp_path, monkeypatch):
    env, out = _setup(
        tmp_path, ["st-1", "st-2", "st-3"],
        [("st-1", "TRAIN"), ("st-2", "TRAIN"), ("st-3", "TRAIN")],
        canonical_rows=[("st-1", "TRAIN"), ("st-2", "TRAIN"), ("st-3", "TRAIN")])
    _apply_env(monkeypatch, env)
    run(["build-sft", "--limit", "1"])
    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 1


def test_rl_pool_is_train_only(tmp_path, monkeypatch):
    # Force explicit canonical train/val/test assignments so the split is deterministic.
    # Order matters: both train records sit at the END (indices 2, 3), with val/test
    # occupying indices 0/1. This makes the test fail if `train_idx` remapping is
    # dropped -- an unmapped subset-index (which lives in [0, len(train)-1], i.e. {0,1})
    # would incorrectly resolve to the val/test record at that position instead of the
    # actual train record. Verified empirically: with the remap line removed, this test
    # fails (pool points at st-test); with it restored, it passes.
    env, out = _setup(
        tmp_path, ["st-val", "st-test", "st-train-a", "st-train-b"],
        [("st-val", "TRAIN"), ("st-test", "TRAIN"), ("st-train-a", "TRAIN"), ("st-train-b", "TRAIN")],
        canonical_rows=[("st-val", "VAL"), ("st-test", "TEST"),
                        ("st-train-a", "TRAIN"), ("st-train-b", "TRAIN")])
    _apply_env(monkeypatch, env)
    run(["build-rl", "--split", "all"])

    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    splits_by_uuid = {r["study_uuid"]: r["split"] for r in rl}
    assert splits_by_uuid == {
        "st-val": "val", "st-test": "test", "st-train-a": "train", "st-train-b": "train"}

    pool = json.load(open(out / "rl_pool.json"))
    indices = pool["indices"]
    assert len(indices) > 0
    for idx in indices:
        assert rl[idx]["split"] == "train"


def test_build_default_split_excludes_val_test(tmp_path, monkeypatch):
    # One TRAIN study, one VAL study, one TEST study -- default --split train must write
    # only the TRAIN study's records for both build-sft and build-rl.
    env, out = _setup(
        tmp_path, ["st-train", "st-val", "st-test"],
        [("st-train", "TRAIN"), ("st-val", "TRAIN"), ("st-test", "TRAIN")],
        canonical_rows=[("st-train", "TRAIN"), ("st-val", "VAL"), ("st-test", "TEST")])
    _apply_env(monkeypatch, env)
    run(["build-sft"])
    run(["build-rl"])

    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 1
    assert {r["study_uuid"] for r in sft} == {"st-train"}

    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert len(rl) == 1
    assert rl[0]["study_uuid"] == "st-train"
    assert rl[0]["split"] == "train"
    assert all(r["split"] not in {"val", "test"} for r in rl)


def test_build_rl_split_all_includes_val_test(tmp_path, monkeypatch):
    env, out = _setup(
        tmp_path, ["st-train", "st-val", "st-test"],
        [("st-train", "TRAIN"), ("st-val", "TRAIN"), ("st-test", "TRAIN")],
        canonical_rows=[("st-train", "TRAIN"), ("st-val", "VAL"), ("st-test", "TEST")])
    _apply_env(monkeypatch, env)
    run(["build-rl", "--split", "all"])

    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert len(rl) == 3
    splits = {r["study_uuid"]: r["split"] for r in rl}
    assert splits == {"st-train": "train", "st-val": "val", "st-test": "test"}

    # rl_pool.json indices must resolve only to split == "train" records.
    pool = json.load(open(out / "rl_pool.json"))
    for idx in pool["indices"]:
        assert rl[idx]["split"] == "train"


def test_build_eval_defaults_to_test_split(monkeypatch, tmp_path):
    """build-eval must not inherit the train default, or it emits zero records."""
    from echo_rl import cli
    seen = {}
    monkeypatch.setattr(cli, "build_eval", lambda cfg, limit, split: seen.update(split=split))
    monkeypatch.setattr(cli.Config, "from_env", classmethod(lambda c: None))
    cli.run(["build-eval"])
    assert seen["split"] == "test"


def test_explicit_split_still_wins(monkeypatch):
    from echo_rl import cli
    seen = {}
    monkeypatch.setattr(cli, "build_eval", lambda cfg, limit, split: seen.update(split=split))
    monkeypatch.setattr(cli.Config, "from_env", classmethod(lambda c: None))
    cli.run(["build-eval", "--split", "val"])
    assert seen["split"] == "val"


def test_build_sft_still_defaults_to_train(monkeypatch):
    from echo_rl import cli
    seen = {}
    monkeypatch.setattr(cli, "build_sft", lambda cfg, limit, split: seen.update(split=split))
    monkeypatch.setattr(cli.Config, "from_env", classmethod(lambda c: None))
    cli.run(["build-sft"])
    assert seen["split"] == "train"
