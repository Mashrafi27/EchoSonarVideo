"""Scorer tests. The theme: a model that fails must LOOK like it failed."""
import pytest

from echo_verl.eval.score_eval import score


def _yn(target, answer, tools=()):
    return {"question_type": "abnormality_classification",
            "reward_key": {"kind": "yesno", "target": target},
            "answer": answer, "gold_answer": target, "tool_calls": list(tools),
            "malformed_tool_calls": 0, "finish_reason": "answered"}


def test_unparsable_answers_count_as_wrong_not_dropped():
    """The dangerous alternative: dropping them lets a model that never answers
    score 100% on the handful it did answer."""
    eps = [_yn("yes", "I cannot tell."), _yn("yes", "Yes."), _yn("no", "No.")]
    r = score(eps)["by_question_type"]["abnormality_classification"]
    assert r["n"] == 3                      # all three scored
    assert r["unparsable"] == 1
    assert r["accuracy"] == pytest.approx(2 / 3)


def test_always_no_is_exposed_by_balanced_accuracy():
    eps = [_yn("no", "No.")] * 8 + [_yn("yes", "No.")] * 2
    r = score(eps)["by_question_type"]["abnormality_classification"]
    assert r["accuracy"] == pytest.approx(0.8)
    assert r["balanced_accuracy"] == pytest.approx(0.5)
    assert r["majority_baseline"]["accuracy"] == pytest.approx(0.8)


def test_bootstrap_ci_is_reported_for_headline_metrics():
    eps = [_yn("no", "No.")] * 10 + [_yn("yes", "Yes.")] * 4
    r = score(eps)["by_question_type"]["abnormality_classification"]
    for k in ("balanced_accuracy_ci", "macro_f1_ci"):
        assert r[k]["ci_lo"] <= r[k]["value"] <= r[k]["ci_hi"]


def test_agentic_metrics_summarise_behaviour():
    eps = [_yn("no", "No.", tools=[{"op": "select_view", "ok": True}]),
           _yn("no", "No."),
           _yn("no", "No.", tools=[{"op": "zoom", "ok": False}, {"op": "select_frames", "ok": True}])]
    a = score(eps)["agentic"]
    assert a["tool_call_rate"] == pytest.approx(2 / 3)
    assert a["total_tool_calls"] == 3
    assert a["failed_tool_calls"] == 1
    assert a["ops"] == {"select_view": 1, "zoom": 1, "select_frames": 1}


def test_no_answer_episodes_are_surfaced():
    eps = [_yn("no", None), _yn("no", "No.")]
    eps[0]["finish_reason"] = "max_turns"
    a = score(eps)["agentic"]
    # Nothing at all to score: no tag AND no untagged output.
    assert a["no_output"] == 1
    assert a["answered_with_tag"] == 1
    assert a["finish_reasons"]["max_turns"] == 1


def test_untagged_output_is_scored_not_discarded():
    """A model that answers correctly WITHOUT <answer> tags must score correctly.

    The tags are a convention our SFT taught. Scoring only tagged text would mark an
    untrained baseline wrong for punctuation rather than for medicine, making any
    trained-vs-untrained comparison meaningless.
    """
    tagged, untagged = _yn("no", "No."), _yn("no", None)
    untagged["final_text"] = "<think>looks unremarkable</think>No, no abnormality is seen."
    report = score([tagged, untagged])
    cls = report["by_question_type"]["abnormality_classification"]
    assert cls["accuracy"] == 1.0, "untagged but correct answer was scored wrong"
    assert cls["unparsable"] == 0
    # ...while tag compliance is still reported honestly as a separate property.
    assert report["agentic"]["answered_with_tag"] == 1
    assert report["agentic"]["no_output"] == 0


def test_reasoning_is_never_scored_as_the_answer():
    """<think> content must not leak into the scored text.

    It is not the answer, and for the NLG metrics it would inflate length and wreck
    BLEU/ROUGE. An unterminated <think> (generation cut off mid-thought) counts as
    no output rather than as a whole-reply answer.
    """
    ep = _yn("yes", None)
    ep["final_text"] = "<think>the left atrium is clearly enlarged, so yes</think>"
    a = score([ep])["agentic"]
    assert a["no_output"] == 1, "bare reasoning was treated as an answer"

    cut = _yn("yes", None)
    cut["final_text"] = "<think>the left atrium is clearly enlarged so the answer is yes"
    assert score([cut])["agentic"]["no_output"] == 1


def test_free_text_reports_nlg_and_coverage():
    gold = "Left Ventricle: Normal.\nRight Ventricle: Normal.\nLeft Atrium: Mild enlargement."
    eps = [{"question_type": "full_report", "reward_key": {"kind": "text", "target": gold},
            "answer": "Left Ventricle: Normal.", "gold_answer": gold,
            "tool_calls": [], "malformed_tool_calls": 0, "finish_reason": "answered"}]
    r = score(eps)["by_question_type"]["full_report"]
    assert r["section_coverage"] == pytest.approx(1 / 3)   # answered 1 of 3 sections
    assert "BLEU-4" in r and "ROUGE-L" in r


def test_empty_answers_are_counted_for_text_types():
    gold = "Left Ventricle: Normal.\nRight Ventricle: Normal."
    eps = [{"question_type": "conclusion", "reward_key": {"kind": "text", "target": gold},
            "answer": "", "gold_answer": gold, "tool_calls": [],
            "malformed_tool_calls": 0, "finish_reason": "max_turns"}]
    r = score(eps)["by_question_type"]["conclusion"]
    assert r["empty_answers"] == 1


def test_absent_question_types_are_absent_from_the_report():
    r = score([_yn("no", "No.")])
    assert "full_report" not in r["by_question_type"]


def test_empty_episode_list_does_not_crash():
    r = score([])
    assert r["n_episodes"] == 0 and r["by_question_type"] == {}
