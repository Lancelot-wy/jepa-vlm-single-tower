import torch

from jepa_vlm.modeling.pooling import SpatialVisualTokenPooler, resolve_pooled_grid


def test_k_sweep_shapes_and_row_major_order():
    x = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8, 1)
    for k, grid in ((4, (2, 2)), (16, (4, 4)), (64, (8, 8))):
        y, actual = SpatialVisualTokenPooler(k)(x)
        assert actual == grid
        assert y.shape == (1, 1, k, 1)
    y64, _ = SpatialVisualTokenPooler(64)(x)
    assert y64.flatten().tolist() == list(range(64))


def test_non_square_grid_keeps_aspect_orientation():
    out_h, out_w = resolve_pooled_grid(16, 4, 16)
    assert out_w >= out_h
    x = torch.randn(2, 3, 4, 16, 7)
    y, grid = SpatialVisualTokenPooler(16)(x)
    assert grid[0] * grid[1] == 16
    assert y.shape == (2, 3, 16, 7)
