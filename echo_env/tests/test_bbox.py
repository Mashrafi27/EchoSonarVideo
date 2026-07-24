from echo_env.bbox import normalize_bbox


def test_valid_bbox_passthrough():
    assert normalize_bbox([10, 20, 100, 200], 336, 336) == (10, 20, 100, 200)


def test_clamp_to_bounds():
    assert normalize_bbox([-5, -5, 500, 500], 336, 336) == (0, 0, 336, 336)


def test_inverted_bbox_rejected():
    assert normalize_bbox([100, 100, 50, 50], 336, 336) is None


def test_tiny_bbox_expanded_to_min_side():
    out = normalize_bbox([160, 160, 170, 170], 336, 336, min_side=32)
    assert out is not None
    left, top, right, bottom = out
    assert (right - left) >= 32 and (bottom - top) >= 32


def test_corner_bbox_below_floor_rejected():
    # A tiny box pinned to the top-left corner cannot expand symmetrically to reach
    # min_side: [0,0,4,4] center-expands but clamps at the edge to (0,0,16,16), 16<32.
    # It must be rejected (None), not passed through below the patch floor.
    assert normalize_bbox([0, 0, 4, 4], 336, 336, min_side=32) is None


def test_edge_bbox_below_floor_rejected():
    # Pinned to the left edge only: height can expand, width cannot reach 32.
    assert normalize_bbox([0, 160, 6, 168], 336, 336, min_side=32) is None


def test_extreme_aspect_ratio_rejected():
    # 300 wide x 2 tall -> aspect 150 > 100, rejected by the FIRST _validate before any
    # min-side expansion runs (the image is only 2px tall, so it also cannot be clamped larger).
    out = normalize_bbox([0, 0, 300, 2], 300, 2, min_side=28, max_aspect=100.0)
    assert out is None


def test_non_numeric_bbox_rejected():
    assert normalize_bbox(["a", "b", "c", "d"], 336, 336) is None
    assert normalize_bbox([1, 2, 3], 336, 336) is None
