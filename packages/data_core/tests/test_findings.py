from data_core.reward.findings import (
    CANONICAL_FINDINGS, extract_canonical_findings, iou, FINDING_TO_DETR_CLASS)


def test_canonical_findings_has_eleven_entries():
    assert len(CANONICAL_FINDINGS) == 11


def test_extract_canonical_findings_matches_aliases():
    text = "- TR\n- mitral regurgitation\n- LV enlargement"
    found = extract_canonical_findings(text)
    assert found == {"tricuspid regurgitation", "mitral valve regurgitation",
                      "left ventricular enlargement"}


def test_extract_canonical_findings_no_abnormalities_returns_empty():
    assert extract_canonical_findings("No significant abnormalities identified.") == set()


def test_extract_canonical_findings_ignores_unrecognized_bullets():
    # a bullet that isn't one of the 11 canonical findings (or an alias) is dropped, not
    # kept as an open free-text entry -- this is the whole point of the closed taxonomy
    text = "- some made up finding\n- aortic stenosis"
    assert extract_canonical_findings(text) == {"aortic stenosis"}


def test_iou_edges():
    assert iou(set(), set()) == 1.0
    assert iou({"a"}, set()) == 0.0
    assert iou(set(), {"a"}) == 0.0
    assert iou({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(iou({"a", "b"}, {"a", "c"}) - 1 / 3) < 1e-9


def test_finding_to_detr_class_covers_every_canonical_finding():
    assert set(FINDING_TO_DETR_CLASS.keys()) == CANONICAL_FINDINGS


def test_finding_to_detr_class_maps_covered_structures():
    assert FINDING_TO_DETR_CLASS["left ventricular enlargement"] == 0
    assert FINDING_TO_DETR_CLASS["right ventricular enlargement"] == 3
    assert FINDING_TO_DETR_CLASS["mitral valve calcification"] == 4
    assert FINDING_TO_DETR_CLASS["mitral valve regurgitation"] == 4
    assert FINDING_TO_DETR_CLASS["tricuspid regurgitation"] == 5
    assert FINDING_TO_DETR_CLASS["left atrial enlargement"] == 1
    assert FINDING_TO_DETR_CLASS["right atrial enlargement"] == 2
    assert FINDING_TO_DETR_CLASS["left ventricular systolic function"] == 0


def test_finding_to_detr_class_none_for_uncovered_aortic_valve_findings():
    # RT-DETR's 7 classes (report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES) have
    # no aortic valve class at all
    assert FINDING_TO_DETR_CLASS["aortic regurgitation"] is None
    assert FINDING_TO_DETR_CLASS["aortic stenosis"] is None
    assert FINDING_TO_DETR_CLASS["bicuspid aortic valve"] is None
