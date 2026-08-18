"""Metric tests, including agreement with sklearn where it is installed.

The agreement tests are the point: these are the numbers we will report next to
CardioBench's, so a stdlib reimplementation that quietly disagrees would make the
comparison meaningless.
"""
import pytest

from echo_rl.eval.metrics import (accuracy, balanced_accuracy, bootstrap_ci, mae,
                                  macro_f1, majority_baseline, per_class_f1, set_f1,
                                  tool_call_rate, tools_per_episode, view_hit_rate)

# deliberately imbalanced, like our eval set
Y_TRUE = ["no"] * 8 + ["yes"] * 2
Y_ALWAYS_NO = ["no"] * 10
Y_MIXED = ["no"] * 7 + ["yes"] + ["no"] + ["yes"]


def test_accuracy_rewards_the_degenerate_model():
    assert accuracy(Y_TRUE, Y_ALWAYS_NO) == 0.8


def test_balanced_accuracy_exposes_it():
    # 100% recall on "no", 0% on "yes" -> 0.5, the coin-flip floor
    assert balanced_accuracy(Y_TRUE, Y_ALWAYS_NO) == 0.5


def test_macro_f1_also_exposes_it():
    # "yes" F1 is 0, so the macro mean is halved
    assert macro_f1(Y_TRUE, Y_ALWAYS_NO) == pytest.approx(0.4444, abs=1e-4)


def test_perfect_prediction_scores_one():
    for fn in (accuracy, balanced_accuracy, macro_f1):
        assert fn(Y_TRUE, list(Y_TRUE)) == 1.0


def test_per_class_f1_keys_come_from_truth_only():
    y_true = ["no"] * 4
    y_pred = ["no", "no", "maybe", "maybe"]
    assert set(per_class_f1(y_true, y_pred)) == {"no"}


def test_majority_baseline_reports_the_floor():
    b = majority_baseline(Y_TRUE)
    assert b["class"] == "no" and b["accuracy"] == 0.8 and b["balanced_accuracy"] == 0.5


def test_empty_inputs_do_not_crash():
    assert accuracy([], []) == 0.0
    assert balanced_accuracy([], []) == 0.0
    assert macro_f1([], []) == 0.0
    assert majority_baseline([])["class"] is None


# ---- set F1 ----

def test_set_f1_partial_credit():
    assert set_f1({"a", "b"}, {"a", "b", "c"}) == pytest.approx(0.8)


def test_set_f1_both_empty_is_correct():
    # correctly reporting no findings is a right answer, not a division by zero
    assert set_f1(set(), set()) == 1.0


def test_set_f1_penalises_invention():
    assert set_f1({"a", "x", "y"}, {"a"}) == pytest.approx(0.5)


def test_set_f1_no_overlap():
    assert set_f1({"x"}, {"a"}) == 0.0


# ---- MAE ----

def test_mae_skips_unparsable_and_returns_none_when_nothing_left():
    assert mae([1.0, 2.0, 3.0], [1.5, None, 3.0]) == pytest.approx(0.25)
    assert mae([1.0], [None]) is None       # never 0.0 -- that would look perfect


# ---- bootstrap ----

def test_bootstrap_ci_brackets_the_point_estimate():
    y_true = ["no"] * 40 + ["yes"] * 10
    y_pred = ["no"] * 40 + ["yes"] * 7 + ["no"] * 3
    out = bootstrap_ci(y_true, y_pred, balanced_accuracy, n=200, seed=42)
    assert out["ci_lo"] <= out["value"] <= out["ci_hi"]
    assert out["n"] == 50


def test_bootstrap_is_deterministic_under_seed():
    a = bootstrap_ci(Y_TRUE, Y_MIXED, macro_f1, n=100, seed=42)
    b = bootstrap_ci(Y_TRUE, Y_MIXED, macro_f1, n=100, seed=42)
    assert a == b


# ---- agentic ----

def test_tool_metrics():
    traces = [[{"op": "select_view"}], [], [{"op": "select_view"}, {"op": "zoom"}]]
    assert tool_call_rate(traces) == pytest.approx(2 / 3)
    assert tools_per_episode(traces) == 1.0


def test_view_hit_rate_counts_only_episodes_with_a_known_target():
    traces = [[{"op": "select_view", "view_name": "A4C"}],
              [{"op": "select_view", "view_name": "PLAX"}],
              [{"op": "select_view", "view_name": "A2C"}]]
    relevant = [["A4C"], ["A4C"], []]      # third has no known relevant view
    assert view_hit_rate(traces, relevant) == 0.5      # 1 of the 2 scorable


def test_view_hit_rate_none_when_nothing_scorable():
    assert view_hit_rate([[{"op": "select_view", "view_name": "A4C"}]], [[]]) is None


def test_zoom_alone_is_not_a_view_hit():
    traces = [[{"op": "zoom", "view_name": "A4C"}]]
    assert view_hit_rate(traces, [["A4C"]]) == 0.0


# ---- agreement with sklearn ----

def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    cases = [(Y_TRUE, Y_ALWAYS_NO), (Y_TRUE, Y_MIXED), (Y_TRUE, list(Y_TRUE))]
    for y_true, y_pred in cases:
        assert accuracy(y_true, y_pred) == pytest.approx(sk.accuracy_score(y_true, y_pred))
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(
            sk.balanced_accuracy_score(y_true, y_pred))
        assert macro_f1(y_true, y_pred) == pytest.approx(
            sk.f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro"))
