"""Bbox-overlap grounding between a model's zoom/select_frames tool calls and RT-DETR-detected
structure boxes, for the abnormality_list tool_bonus (spec: docs/superpowers/specs/
2026-09-15-echoprime-tool-grpo-design.md section 5, item "set"). Pure math only -- no h5/file
I/O here (verl_bridge/reward.py, which already does process-level state, owns opening the h5
and the frame_dims lookup table and passes plain arrays/tuples in).

DETR's boxes_xyxy are real pixel coordinates (confirmed this session by direct inspection:
sample values up to ~374, not [0,1]) -- normalize_detr_box needs the frame's real pixel
dimensions. We now self-run RT-DETR on our own uniformly-336x336 preprocessed frames
(`packages/echoprime_track/build_detr_cache.py`, on the raw PNG via `load_clip_frames`, no
cropping), so boxes always come back in the known fixed (336, 336) pixel space -- no
per-dicom frame-dimension lookup/precompute needed (the earlier `scripts/build_frame_dims.py`
this docstring used to reference has been deleted; see
docs/superpowers/plans/2026-09-16-echoprime-self-inference.md Task 6).
"""


def bbox_iou(a: tuple, b: tuple) -> float:
    """Both boxes (left, top, right, bottom), same coordinate space (normalized [0,1] here)."""
    al, at, ar, ab = a
    bl, bt, br, bb = b
    inter_l = max(al, bl)
    inter_t = max(at, bt)
    inter_r = min(ar, br)
    inter_b = min(ab, bb)
    inter_w = max(0.0, inter_r - inter_l)
    inter_h = max(0.0, inter_b - inter_t)
    inter = inter_w * inter_h
    area_a = max(0.0, ar - al) * max(0.0, ab - at)
    area_b = max(0.0, br - bl) * max(0.0, bb - bt)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_detr_box(box_xyxy: tuple, frame_width: int, frame_height: int) -> tuple:
    left, top, right, bottom = box_xyxy
    return (left / frame_width, top / frame_height, right / frame_width, bottom / frame_height)


def ground_tool_calls(tool_calls: list, view: str, detr_boxes_norm: list) -> float:
    """`tool_calls`: list of {"name": ..., "arguments": {...}} dicts (tool_env.parse.parse_action's
    shape) already filtered to calls referencing `view`. `detr_boxes_norm`: list of already-
    normalized (left, top, right, bottom) DETR boxes for the relevant class, across whichever
    frames matter for this call. Returns the BEST (max) IoU across every call x box pair --
    "did the model ever point at roughly the right place", not an average (a model that zooms
    once correctly after one bad guess shouldn't be penalized for the bad guess).

    select_frames calls have no bbox (only frame_indices) -- treated as the whole-frame box
    (0, 0, 1, 1), same normalized convention grid.py uses elsewhere in this project. This gives
    select_frames calls touching the right view partial (low) credit even without spatial
    precision, while zoom calls that actually bound the structure score much higher."""
    if not tool_calls or not detr_boxes_norm:
        return 0.0
    best = 0.0
    for call in tool_calls:
        if call.get("name") == "zoom":
            bbox = tuple(call.get("arguments", {}).get("bbox", ()))
            if len(bbox) != 4:
                continue
        elif call.get("name") == "select_frames":
            bbox = (0.0, 0.0, 1.0, 1.0)
        else:
            continue
        for detr_box in detr_boxes_norm:
            best = max(best, bbox_iou(bbox, detr_box))
    return best
