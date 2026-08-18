"""NLG metric tests, including agreement with the reference implementations.

The agreement tests matter more than the unit tests here: these numbers go next
to EchoSonar-R's BLEU-4 0.725 / ROUGE-L 0.819, so a stdlib version that quietly
disagrees with sacrebleu/rouge_score would make the comparison meaningless.
"""
import pytest

from echo_rl.eval.nlg import bleu, bleu_n, nlg_report, rouge_l, rouge_l_corpus, tokenize

GOLD = "Normal left ventricular size and ejection fraction with mild mitral regurgitation."


def test_identical_text_scores_one():
    out = nlg_report([GOLD], [GOLD])
    assert out["BLEU-4"] == pytest.approx(1.0)
    assert out["ROUGE-L"] == pytest.approx(1.0)


def test_unrelated_text_scores_zero():
    out = nlg_report(["The weather is fine today."], [GOLD])
    assert out["BLEU-4"] == 0.0
    assert out["ROUGE-L"] == 0.0


def test_bleu_orders_partial_matches_sensibly():
    close = "Normal left ventricular size and ejection fraction with mild mitral regurgitation seen."
    far = "Severe dilation with reduced function."
    assert bleu([close], [GOLD]) > bleu([far], [GOLD])


def test_brevity_penalty_punishes_truncation():
    """A short prefix has perfect precision; only the brevity penalty stops it
    from scoring 1.0."""
    assert bleu(["Normal left ventricular size"], [GOLD]) < 1.0


def test_repetition_is_clipped():
    # padding with a repeated reference word must not inflate precision
    spam = "normal normal normal normal normal normal normal normal"
    assert bleu([spam], [GOLD]) < 0.2


def test_rouge_l_is_recall_biased():
    """beta=1.2 weights recall above precision: omitting content should cost more
    than adding it."""
    half = "Normal left ventricular size and ejection fraction"
    padded = GOLD + " No pericardial effusion was seen on this study."
    assert rouge_l(padded, GOLD) > rouge_l(half, GOLD)


def test_empty_inputs_do_not_crash():
    assert bleu([], []) == 0.0
    assert rouge_l("", GOLD) == 0.0
    assert rouge_l(GOLD, "") == 0.0
    assert rouge_l_corpus([], []) == 0.0


def test_tokenizer_strips_punctuation_and_case():
    assert tokenize("Mild MR, mild TR.") == ["mild", "mr", "mild", "tr"]


def test_bleu_n_reports_all_four_orders():
    out = bleu_n([GOLD], [GOLD])
    assert set(out) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"}


def test_absent_model_metrics_are_absent_not_zero():
    """BERTScore/METEOR/GREEN need models. A silent 0.0 would read as a real
    measurement of a bad model."""
    out = nlg_report([GOLD], [GOLD])
    for absent in ("BERTScore", "METEOR", "GREEN"):
        assert absent not in out


# ---- agreement with reference implementations ----

def test_bleu_matches_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    preds = [GOLD, "Mild mitral regurgitation.", "Severe dilation with reduced function."]
    refs = [GOLD, "Mild mitral regurgitation noted.", GOLD]
    theirs = sacrebleu.corpus_bleu(
        preds, [refs], tokenize="none", force=True,
        smooth_method="none", lowercase=True).score / 100.0
    # our tokenizer strips punctuation, theirs is told not to tokenize, so compare
    # on pre-tokenized text to isolate the BLEU arithmetic itself
    preds_t = [" ".join(tokenize(p)) for p in preds]
    refs_t = [" ".join(tokenize(r)) for r in refs]
    theirs = sacrebleu.corpus_bleu(
        preds_t, [refs_t], tokenize="none", force=True,
        smooth_method="none", lowercase=True).score / 100.0
    assert bleu(preds, refs) == pytest.approx(theirs, abs=1e-6)


def test_rouge_l_matches_rouge_score():
    rs = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    for pred, ref in [(GOLD, GOLD),
                      ("Mild mitral regurgitation.", GOLD),
                      ("Normal left ventricular size", GOLD)]:
        theirs = scorer.score(" ".join(tokenize(ref)), " ".join(tokenize(pred)))["rougeL"].fmeasure
        # rouge_score uses beta=1 (plain F1); ours defaults to the ROUGE package's 1.2
        assert rouge_l(pred, ref, beta=1.0) == pytest.approx(theirs, abs=1e-6)
