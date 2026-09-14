from math import ceil, floor


def _validate(left, top, right, bottom, max_aspect) -> bool:
    if not (left < right and top < bottom):
        return False
    h = bottom - top
    w = right - left
    if max(h, w) / min(h, w) > max_aspect:
        return False
    return True


def normalize_bbox(bbox, width, height, min_side=32, max_aspect=100.0):
    """Clamp to [0,0,width,height], expand sub-min_side sides around the center,
    validate ordering + aspect ratio, and enforce the min_side floor. Return an
    int tuple or None.

    A box pinned against a frame edge/corner cannot expand symmetrically to reach
    min_side (there is no room to grow outward), so it may stay below the floor
    after clamping. Such boxes are rejected (None) rather than silently passed
    through sub-floor -- a too-small crop is a bad request, better to make the
    caller pick a larger region than to feed the model an undersized/stretched crop."""
    if not hasattr(bbox, "__len__") or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    if not _validate(left, top, right, bottom, max_aspect):
        return None
    h = bottom - top
    w = right - left
    if h < min_side or w < min_side:
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        ratio = min_side / min(h, w)
        half_w = ceil(w * ratio * 0.5)
        half_h = ceil(h * ratio * 0.5)
        left = max(0, floor(cx - half_w))
        right = min(width, ceil(cx + half_w))
        top = max(0, floor(cy - half_h))
        bottom = min(height, ceil(cy + half_h))
        if not _validate(left, top, right, bottom, max_aspect):
            return None
        # Edge/corner clamping can leave a side below the floor even after
        # center-expansion; reject rather than emit a sub-min_side crop.
        if (bottom - top) < min_side or (right - left) < min_side:
            return None
    return (int(left), int(top), int(right), int(bottom))
