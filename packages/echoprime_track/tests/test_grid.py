import pytest

from echoprime_track.grid import (
    resolve_temporal_group, group_token_slice, spatial_subset,
    N_TEMPORAL_GROUPS, TOKENS_PER_GROUP,
)


def test_resolve_temporal_group_maps_frame_pairs():
    # 16 frames -> 8 groups of 2: frame // 2
    assert resolve_temporal_group([0]) == 0
    assert resolve_temporal_group([1]) == 0
    assert resolve_temporal_group([2]) == 1
    assert resolve_temporal_group([15]) == 7


def test_resolve_temporal_group_uses_first_valid_index_when_spanning_groups():
    # spec: cap at 1 group per call -- use the group of the first valid index
    assert resolve_temporal_group([3, 10]) == 1  # frame 3 -> group 1


def test_resolve_temporal_group_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_temporal_group([16])
    with pytest.raises(ValueError):
        resolve_temporal_group([-1])
    with pytest.raises(ValueError):
        resolve_temporal_group([])


def test_group_token_slice_layout():
    # token 0 is the global/CLS token; group 0 is tokens [1, 50), group 7 is [344, 393)
    assert group_token_slice(0) == slice(1, 50)
    assert group_token_slice(7) == slice(344, 393)
    assert (group_token_slice(7).stop - group_token_slice(7).start) == TOKENS_PER_GROUP


def test_group_token_slice_rejects_out_of_range_group():
    with pytest.raises(ValueError):
        group_token_slice(N_TEMPORAL_GROUPS)
    with pytest.raises(ValueError):
        group_token_slice(-1)


def test_spatial_subset_full_bbox_returns_whole_group():
    idxs = spatial_subset(group=0, bbox=(0.0, 0.0, 1.0, 1.0))
    assert idxs == list(range(1, 50))


def test_spatial_subset_partial_bbox_returns_fewer_tokens():
    # top-left quadrant of the 7x7 grid
    idxs = spatial_subset(group=0, bbox=(0.0, 0.0, 0.5, 0.5))
    assert 0 < len(idxs) < 49
    assert all(1 <= i < 50 for i in idxs)


def test_spatial_subset_offsets_by_group():
    idxs_g0 = spatial_subset(group=0, bbox=(0.0, 0.0, 1.0, 1.0))
    idxs_g1 = spatial_subset(group=1, bbox=(0.0, 0.0, 1.0, 1.0))
    assert idxs_g1 == [i + 49 for i in idxs_g0]


def test_spatial_subset_empty_bbox_raises():
    with pytest.raises(ValueError):
        spatial_subset(group=0, bbox=(0.9, 0.9, 0.91, 0.91), grid_side=7)  # no cell center falls inside
