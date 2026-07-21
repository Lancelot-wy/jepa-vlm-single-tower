import numpy as np
import torch

from jepa_vlm.data.temporal_units import (
    build_temporal_units,
    state_pair_indices,
    temporal_diagnostics,
    temporal_unit_frame_ids,
)
from jepa_vlm.data.video_io import patchify


def test_32_numbered_frames_form_16_real_units():
    ids = np.arange(32)
    units = temporal_unit_frame_ids(ids, 2)
    assert units.shape == (16, 2)
    assert units.tolist() == [[i, i + 1] for i in range(0, 32, 2)]
    source, target = state_pair_indices(16, 2)
    assert source[0] == 0 and target[0] == 2
    assert np.all(target - source == 2)
    assert 2 * 2 / 4.0 == 1.0


def test_processor_order_is_not_duplicate_frame_order():
    frames = np.stack([np.full((2, 2, 1), index, np.uint8) for index in range(32)])
    units = build_temporal_units(frames, 2)
    assert units[:, :, 0, 0, 0].reshape(-1).tolist() == list(range(32))
    diagnostics = temporal_diagnostics(np.arange(32), 4.0, 2, 2)
    assert diagnostics.state_eligible
    assert diagnostics.duplicate_adjacent_ratio == 0.0
    assert diagnostics.temporal_unit_count == 16


def test_state_rejects_wrong_effective_fps_even_with_unique_frames():
    diagnostics = temporal_diagnostics(
        np.arange(32), native_fps=8.0, temporal_patch_size=2,
        horizon_units=2, expected_fps=4.0,
    )
    assert diagnostics.effective_fps == 8.0
    assert not diagnostics.state_eligible


def test_patchify_uses_two_different_real_images_per_temporal_patch():
    # Constant RGB values make temporal order visible in the flattened patch vector.
    frames = torch.stack([torch.full((3, 32, 32), index / 31) for index in range(32)])
    patches, grid = patchify(frames, duplicate_frames=False, temporal_patch_size=2)
    assert tuple(grid.tolist()) == (16, 2, 2)
    first_patch = patches[0]
    half = first_patch.numel() // 2
    assert not torch.allclose(first_patch[:half], first_patch[half:])
