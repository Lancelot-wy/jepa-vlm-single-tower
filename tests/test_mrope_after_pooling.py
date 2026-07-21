import torch

from jepa_vlm.modeling.model import mixed_position_ids, visual_only_position_ids


def test_visual_mrope_shapes_for_all_k():
    for grid in ((2, 2), (4, 4), (8, 8)):
        k = grid[0] * grid[1]
        pos = visual_only_position_ids(2, 16, torch.device("cpu"), grid)
        assert pos.shape == (4, 2, 16 * k)
        rows = pos[2, 0, :k]
        cols = pos[3, 0, :k]
        assert len(set(zip(rows.tolist(), cols.tolist()))) == k


def test_mixed_mrope_matches_placeholder_count():
    k, units = 16, 3
    ids = torch.tensor([[11] + [8] * (k * units) + [12]])
    mask = ids == 8
    pos = mixed_position_ids(ids, mask, units, (4, 4))
    assert pos.shape == (4, 1, ids.shape[1])
