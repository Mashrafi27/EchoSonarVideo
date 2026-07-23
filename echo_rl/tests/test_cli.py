import json, os
from echo_rl.cli import run


def _setup(tmp_path):
    # fake preprocessed_data with one study + one A4C clip
    pp = tmp_path / "pp"; clip = pp / "st-1" / "di-2_A4C"; clip.mkdir(parents=True)
    for i in range(6):
        (clip / f"{i}.png").write_bytes(b"x")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({
        "study_uuid": "st-1", "question_type": "abnormality_classification",
        "thinking": "#### 1. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n",
        "messages": json.dumps([{"role": "user", "content": "Q?"},
                                {"role": "assistant", "content": "No, none."}])}) + "\n")
    test = tmp_path / "test.jsonl"; test.write_text("")
    gold = tmp_path / "gold"; gold.mkdir()
    (gold / "heart_failure_classification.csv").write_text(
        "study_id,label,study designation,from_measurement,from_text,text,text_fields\nst-1,0,TRAIN,T,T,t,c\n")
    out = tmp_path / "out"
    env = {"ECHO_PREPROCESSED_DIR": str(pp), "ECHO_VQA_TRAIN": str(train),
           "ECHO_VQA_TEST": str(test), "ECHO_GOLD_DIR": str(gold), "ECHO_OUT_DIR": str(out)}
    return env, out


def test_build_sft_and_rl(tmp_path, monkeypatch):
    env, out = _setup(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    run(["build-sft"])
    run(["build-rl"])
    sft = [json.loads(l) for l in open(out / "sft.jsonl")]
    assert len(sft) == 1 and sft[0]["trajectory"]["turns"][0]["args"]["view"] == "A4C"
    rl = [json.loads(l) for l in open(out / "rl.jsonl")]
    assert rl[0]["reward_key"]["kind"] == "yesno"
    pool = json.load(open(out / "rl_pool.json"))
    assert isinstance(pool["indices"], list) and len(pool["indices"]) > 0
