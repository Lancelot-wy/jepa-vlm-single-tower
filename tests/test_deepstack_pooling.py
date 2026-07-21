import torch

from jepa_vlm.modeling.pooling import SpatialVisualTokenPooler


def test_main_and_three_deepstack_levels_share_grid_and_k():
    pooler = SpatialVisualTokenPooler(16)
    main = torch.randn(2, 16, 8, 8, 32)
    pooled_main, grid = pooler(main)
    levels = [pooler(torch.randn_like(main), grid)[0] for _ in range(3)]
    assert pooled_main.shape[-2] == 16
    assert all(level.shape == pooled_main.shape for level in levels)
