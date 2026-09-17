from data_core.reward.grounding import bbox_iou, normalize_detr_box, ground_tool_calls


def test_bbox_iou_identical_boxes():
    assert bbox_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0


def test_bbox_iou_disjoint_boxes():
    assert bbox_iou((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)) == 0.0


def test_bbox_iou_partial_overlap():
    # two unit-area boxes overlapping in a 0.5x0.5 region: intersection 0.25, union 1.75
    val = bbox_iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 1.5, 1.5))
    assert abs(val - 0.25 / 1.75) < 1e-9


def test_normalize_detr_box():
    # pixel box (100, 100, 200, 300) in a 400x400 frame -> (0.25, 0.25, 0.5, 0.75)
    out = normalize_detr_box((100.0, 100.0, 200.0, 300.0), frame_width=400, frame_height=400)
    assert out == (0.25, 0.25, 0.5, 0.75)
