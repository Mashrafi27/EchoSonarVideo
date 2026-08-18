"""Evaluation metrics — deliberately SEPARATE from echo_rl.reward.score.

The reward is what the model optimises; these are what we report. Keeping them
in different modules is the point: a number the model was trained to maximise is
not evidence about the model. They share the same underlying notion of a correct
answer, but they are computed, weighted, and aggregated independently.

Definitions follow CardioBench (Taratynova et al., arXiv 2510.00520) so numbers
are comparable with the group's prior echo work:
  evaluation/CardioBench/evaluation/utils.py::classification_metrics
    Accuracy(%)          = 100 * accuracy_score
    Balanced_Accuracy(%) = 100 * balanced_accuracy_score   (mean per-class recall)
    F1(%)                = 100 * f1_score(average="macro") in the bootstrap path
  ::bootstrap_classification  — B=1000 resamples, seed=42, percentile 95% CI
Implemented in stdlib so the offline test venv needs no sklearn; verified against
sklearn where available (see tests).

Balanced accuracy is not optional here. Measured on build/eval.jsonl,
abnormality_classification is 10916 "no" vs 2449 "yes", so a model that always
answers "no" scores 81.7% plain accuracy while being clinically useless.
"""
import random
from collections import Counter


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def balanced_accuracy(y_true: list, y_pred: list) -> float:
    """Mean of per-class recall — sklearn's balanced_accuracy_score."""
    if not y_true:
        return 0.0
    recalls = []
    for cls in sorted(set(y_true)):
        support = [i for i, t in enumerate(y_true) if t == cls]
        if not support:
            continue
        recalls.append(sum(1 for i in support if y_pred[i] == cls) / len(support))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _prf(y_true, y_pred, cls):
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t != cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if p != cls and t == cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def macro_f1(y_true: list, y_pred: list) -> float:
    """Unweighted mean per-class F1 — sklearn f1_score(average='macro').

    Classes come from y_true only, matching CardioBench's labels=np.unique(y_true):
    a class the model hallucinates but that never occurs does not get its own term.
    """
    classes = sorted(set(y_true))
    if not classes:
        return 0.0
    return sum(_prf(y_true, y_pred, c)[2] for c in classes) / len(classes)


def per_class_f1(y_true: list, y_pred: list) -> dict:
    return {c: _prf(y_true, y_pred, c)[2] for c in sorted(set(y_true))}


def set_f1(pred: set, gold: set) -> float:
    """F1 over two sets of finding names — the multi-label case, per sample.

    Both empty counts as 1.0: correctly reporting no findings is correct.
    """
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def mae(y_true: list, y_pred: list) -> float:
    """Mean absolute error over pairs where BOTH values are present.

    Returns None if nothing is comparable, rather than 0.0 — a model that never
    emits a parsable number must not look perfect.
    """
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return None
    return sum(abs(t - p) for t, p in pairs) / len(pairs)


def bootstrap_ci(y_true: list, y_pred: list, metric_fn, *, n: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI. CardioBench defaults: B=1000, seed=42, 95%.

    Returns the point estimate on the full sample plus the resample mean and
    interval, so a wide interval is visible rather than implied.
    """
    point = metric_fn(y_true, y_pred)
    if not y_true or point is None:
        return {"value": point, "mean": None, "ci_lo": None, "ci_hi": None, "n": len(y_true)}
    rng = random.Random(seed)
    size = len(y_true)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        v = metric_fn([y_true[i] for i in idx], [y_pred[i] for i in idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return {"value": point, "mean": None, "ci_lo": None, "ci_hi": None, "n": size}
    vals.sort()
    lo = vals[max(0, int((alpha / 2) * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return {"value": point, "mean": sum(vals) / len(vals),
            "ci_lo": lo, "ci_hi": hi, "n": size}


def majority_baseline(y_true: list) -> dict:
    """What 'always answer the commonest class' scores. Report alongside any
    accuracy, or a useless model looks strong."""
    if not y_true:
        return {"class": None, "accuracy": 0.0, "balanced_accuracy": 0.0}
    cls = Counter(y_true).most_common(1)[0][0]
    y_pred = [cls] * len(y_true)
    return {"class": cls,
            "accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred)}


# ---- agentic metrics (DeepEyes-style; these are what distinguish this work) ----

def tool_call_rate(traces: list) -> float:
    """Fraction of episodes that called the echo tool at least once."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t) / len(traces)


def tools_per_episode(traces: list) -> float:
    if not traces:
        return 0.0
    return sum(len(t) for t in traces) / len(traces)


def view_hit_rate(traces: list, relevant_views: list) -> float:
    """Did the model open a view where the finding is actually visible?

    Our analogue of DeepEyes' grounding IoU: they measure whether the zoomed BOX
    contains the object, we measure whether the opened VIEW contains the finding.
    Episodes with no known relevant view are skipped, not counted as failures.
    """
    scored = [(t, set(r)) for t, r in zip(traces, relevant_views) if r]
    if not scored:
        return None
    hits = 0
    for calls, rel in scored:
        opened = {c.get("view_name") for c in calls if c.get("op") == "select_view"}
        if opened & rel:
            hits += 1
    return hits / len(scored)
