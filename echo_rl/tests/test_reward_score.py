from echo_rl.reward.score import (
    f1, score_yesno, score_set, extract_entities, score_entity_f1,
    score_gold_value, score_outcome, NullJudge,
)


def test_f1_edges():
    assert f1(set(), set()) == 1.0            # both empty = perfect
    assert f1({"a"}, set()) == 0.0
    assert f1(set(), {"a"}) == 0.0
    assert f1({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(f1({"a", "b"}, {"a"}) - 2/3) < 1e-9   # P=1/2, R=1 -> F1=2/3


def test_score_yesno():
    assert score_yesno("Yes, there is dilation.", "yes") == 1.0
    assert score_yesno("No.", "no") == 1.0
    assert score_yesno("Yes", "no") == 0.0
    assert score_yesno("Maybe", "yes") == 0.0   # parse_yes_no -> None


def test_score_set():
    ans = "- mitral regurgitation\n- lv dilation"
    assert score_set(ans, ["mitral regurgitation", "lv dilation"]) == 1.0
    assert score_set("No significant abnormalities.", []) == 1.0
    assert 0.0 < score_set(ans, ["mitral regurgitation"]) < 1.0


def test_extract_entities_and_entity_f1():
    pred = "The LV is dilated with reduced systolic function."
    ref = "Dilated LV, reduced function."
    ents = extract_entities(pred)
    assert "dilat" in ents and "reduced" in ents
    assert score_entity_f1(pred, ref) > 0.0
    assert score_entity_f1("Totally normal study.", "Severe stenosis and dilation.") == 0.0


def test_score_gold_value():
    assert score_gold_value("EF is severely reduced.", {"ef": "severely reduced"}) == 1.0
    assert score_gold_value("Normal.", {"ef": "severely reduced"}) == 0.0
    assert score_gold_value("anything", {}) is None       # nothing to score


def test_score_outcome_dispatch():
    assert score_outcome({"kind": "yesno", "target": "yes", "gold": {}}, "Yes.") == 1.0
    assert score_outcome({"kind": "set", "target": ["lv dilation"], "gold": {}},
                         "- lv dilation") == 1.0
    # text: no gold, NullJudge -> falls back to entity-F1
    r = score_outcome({"kind": "text", "target": "Dilated LV.", "gold": {}},
                      "The LV is dilated.", judge=NullJudge())
    assert r > 0.0


def test_score_outcome_text_prefers_gold_when_present():
    rk = {"kind": "text", "target": "long free text", "gold": {"ef": "reduced"}}
    assert score_outcome(rk, "EF reduced.") == 1.0
    assert score_outcome(rk, "EF normal.") == 0.0
