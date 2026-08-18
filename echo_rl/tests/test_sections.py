from echo_rl.reward.sections import (score_by_section, section_coverage,
                                     split_sections, split_sentences)

REPORT = ("Left Ventricle: Normal dimensions and systolic function.\n"
          "Right Ventricle: Mildly dilated.\n"
          "Left Atrium: Mild enlargement.\n"
          "Conclusions: Overall normal study.")

EXACT = lambda p, g: 1.0 if p.strip() == g.strip() else 0.0


def test_splits_labelled_sections():
    s = split_sections(REPORT)
    assert list(s) == ["left ventricle", "right ventricle", "left atrium", "conclusions"]
    assert s["right ventricle"] == "Mildly dilated."


def test_wrapped_continuation_lines_join_their_section():
    # measured in the real data: non-header lines are wrapped continuations
    text = "Left Ventricle: Normal dimensions and no wall motion\nabnormalities.\nRight Ventricle: Normal."
    s = split_sections(text)
    assert s["left ventricle"] == "Normal dimensions and no wall motion abnormalities."
    assert s["right ventricle"] == "Normal."


def test_unstructured_text_yields_no_sections():
    assert split_sections("Normal dimensions. Normal Doppler flow profile.") == {}
    assert split_sections("") == {}


def test_labels_are_case_and_space_normalised():
    assert "left ventricle" in split_sections("Left  Ventricle:  Normal.")


def test_prose_colon_is_not_a_header():
    # a long mid-sentence clause must not be mistaken for a section label
    long_clause = "The following findings were noted in this particular study today: normal."
    assert split_sections(long_clause) == {}


# ---- scoring ----

def test_identical_report_scores_one():
    assert score_by_section(REPORT, REPORT, EXACT) == 1.0


def test_partial_credit_is_per_section():
    pred = REPORT.replace("Mildly dilated.", "Severely dilated.")
    assert score_by_section(pred, REPORT, EXACT) == 0.75      # 3 of 4 sections


def test_omitted_section_scores_zero_not_skipped():
    """The bug this module exists to prevent: answering one section perfectly
    must not score 1.0."""
    pred = "Left Ventricle: Normal dimensions and systolic function."
    assert score_by_section(pred, REPORT, EXACT) == 0.25      # 1 of 4, not 1.0


def test_wrong_section_wrong_place_does_not_earn_credit():
    # right content under the wrong label -- alignment is by label, deliberately
    pred = ("Left Ventricle: Mildly dilated.\n"
            "Right Ventricle: Normal dimensions and systolic function.\n"
            "Left Atrium: Mild enlargement.\nConclusions: Overall normal study.")
    assert score_by_section(pred, REPORT, EXACT) == 0.5


def test_extra_invented_sections_are_ignored():
    pred = REPORT + "\nSpleen: Unremarkable."
    assert score_by_section(pred, REPORT, EXACT) == 1.0


def test_unstructured_gold_returns_none_for_whole_text_fallback():
    assert score_by_section("anything", "Normal dimensions. Normal flow.", EXACT) is None


def test_single_section_gold_returns_none():
    assert score_by_section("Left Ventricle: Normal.", "Left Ventricle: Normal.", EXACT) is None


# ---- coverage ----

def test_coverage_distinguishes_thorough_from_narrow():
    narrow = "Left Ventricle: Normal dimensions and systolic function."
    assert section_coverage(narrow, REPORT) == 0.25
    assert section_coverage(REPORT, REPORT) == 1.0


def test_coverage_of_unstructured_gold_is_zero():
    assert section_coverage("x", "no headers here") == 0.0


# ---- sentences ----

def test_sentence_split():
    s = split_sentences("Normal LV size. Mild MR. Trace TR.")
    assert s == ["Normal LV size.", "Mild MR.", "Trace TR."]


def test_sentence_split_handles_empty():
    assert split_sentences("") == []
