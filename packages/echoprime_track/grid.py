"""Pure indexing over the precomputed (393, 768) per-view clip-token grid (Darya's
report_generation/data/clip_tokens_{train,test}.h5, confirmed this session: 393 = 1 global/CLS
token + 8 temporal groups (16 frames -> 8 groups of 2) x 49 spatial tokens (7x7 grid)).
select_frames/zoom (echoprime_tool_agent_loop.py) use these to resolve a model tool call into
absolute token indices -- no live EchoPrime encoder inference, ever (spec: docs/superpowers/
specs/2026-09-15-echoprime-tool-grpo-design.md section 2).
"""
import math

N_TEMPORAL_GROUPS = 8
FRAMES_PER_GROUP = 2
TOKENS_PER_GROUP = 49
GRID_SIDE = 7
N_FRAMES = N_TEMPORAL_GROUPS * FRAMES_PER_GROUP  # 16
GLOBAL_TOKEN_COUNT = 1  # index 0


def resolve_temporal_group(frame_indices: list) -> int:
    """First valid frame index's temporal group (0..7). Capped at 1 group per tool call
    (spec's "Token budget" decision) -- callers with indices spanning multiple groups get the
    first one; the caller (echoprime_tool_agent_loop.py) is responsible for telling the model
    that in the tool response text."""
    valid = [i for i in frame_indices if isinstance(i, int) and 0 <= i < N_FRAMES]
    if not valid:
        raise ValueError(f"no valid frame indices in {frame_indices!r} (need 0..{N_FRAMES - 1})")
    return valid[0] // FRAMES_PER_GROUP


def group_token_slice(group: int) -> slice:
    """Absolute [start, stop) token-index slice into the 393-token grid for one temporal
    group. Group 0 starts at index 1 (index 0 is the global/CLS token)."""
    if not (0 <= group < N_TEMPORAL_GROUPS):
        raise ValueError(f"group {group} out of range 0..{N_TEMPORAL_GROUPS - 1}")
    start = GLOBAL_TOKEN_COUNT + group * TOKENS_PER_GROUP
    return slice(start, start + TOKENS_PER_GROUP)


def spatial_subset(group: int, bbox, grid_side: int = GRID_SIDE) -> list:
    """Absolute token indices within `group`'s 49-token block whose spatial cell CENTER falls
    inside `bbox` (left, top, right, bottom), each in [0, 1] normalized coordinates over the
    view's frame -- same normalized convention as tool_env/bbox.py's callers elsewhere in this
    project, for consistency. Raises ValueError if no cell center falls inside (an empty
    selection is a bad tool call, not silently a no-op observation)."""
    left, top, right, bottom = bbox
    if not (left < right and top < bottom):
        raise ValueError(f"invalid bbox {bbox!r}: left<right and top<bottom required")
    base = group_token_slice(group).start
    out = []
    for row in range(grid_side):
        cy = (row + 0.5) / grid_side
        if not (top <= cy <= bottom):
            continue
        for col in range(grid_side):
            cx = (col + 0.5) / grid_side
            if left <= cx <= right:
                out.append(base + row * grid_side + col)
    if not out:
        raise ValueError(f"bbox {bbox!r} contains no grid cell center in group {group}")
    return out
