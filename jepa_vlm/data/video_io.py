"""Video decoding (PyAV), frame sampling, and Qwen3-VL patchification.

The patchify math is copied verbatim from
transformers/models/qwen3_vl/video_processing_qwen3_vl.py (_preprocess) so the
patch layout is guaranteed to match what Qwen3VLVisionModel expects.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .temporal_units import temporal_diagnostics

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


# ---------------------------------------------------------------------- decoding
def _sample_indices(
    total_frames: int,
    native_fps: float,
    num_frames: int,
    sample_fps: float,
    mode: str,
    random_offset: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if mode == "uniform" or sample_fps <= 0:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).round().astype(int)
    # fps_or_uniform: sample num_frames at sample_fps; fall back to uniform on short clips
    step = native_fps / sample_fps
    span = step * (num_frames - 1) + 1
    if span >= total_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).round().astype(int)
    max_start = total_frames - span
    if random_offset and rng is not None:
        start = float(rng.uniform(0, max_start))
    else:
        start = max_start / 2
    return (start + np.arange(num_frames) * step).round().astype(int)


def decode_frames(
    path: str,
    num_frames: int,
    sample_fps: float = 2.0,
    sampling: str = "fps_or_uniform",
    start: float | None = None,
    end: float | None = None,
    random_offset: bool = False,
    rng: np.random.Generator | None = None,
    return_metadata: bool = False,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
    state_horizon_units: int = 1,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Decode `num_frames` frames as uint8 (T, H, W, 3). `start`/`end` (seconds) crop a segment."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or 24.0)
        total = stream.frames
        if not total:  # some containers (webm) don't expose frame count
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            if duration is None and container.duration:
                duration = container.duration / av.time_base
            total = int((duration or 10.0) * native_fps)

        lo = int((start or 0) * native_fps)
        hi = min(int(end * native_fps), total) if end else total
        seg_total = max(hi - lo, 1)
        idx = _sample_indices(seg_total, native_fps, num_frames, sample_fps, sampling, random_offset, rng) + lo
        wanted = set(int(i) for i in idx)

        frames: dict[int, np.ndarray] = {}
        first = min(wanted)
        if first > 5 * native_fps:  # long skip: seek near the first wanted frame
            container.seek(int(first / native_fps / stream.time_base), stream=stream)
        pos_hint = None
        for frame in container.decode(video=0):
            if frame.pts is not None:
                pos_hint = int(round(float(frame.pts * stream.time_base) * native_fps))
            else:
                pos_hint = 0 if pos_hint is None else pos_hint + 1
            if pos_hint in wanted and pos_hint not in frames:
                frames[pos_hint] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(wanted) or pos_hint > max(wanted):
                break

    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    # fill any missing indices with the nearest decoded frame
    keys = sorted(frames)
    out = []
    resolved_idx = []
    for i in idx:
        i = int(i)
        k = i if i in frames else min(keys, key=lambda x: abs(x - i))
        out.append(frames[k])
        resolved_idx.append(k)
    result = np.stack(out)
    if not return_metadata:
        return result
    diagnostics = temporal_diagnostics(
        np.asarray(resolved_idx), native_fps=native_fps,
        temporal_patch_size=temporal_patch_size,
        horizon_units=state_horizon_units,
        expected_fps=sample_fps,
    ).to_dict()
    diagnostics["sampled_frame_ids"] = [int(value) for value in idx]
    diagnostics["decoded_frame_ids"] = [int(value) for value in resolved_idx]
    diagnostics["nearest_frame_substitutions"] = int(
        np.count_nonzero(np.asarray(resolved_idx) != idx)
    )
    diagnostics["native_fps"] = float(native_fps)
    return result, diagnostics


# ---------------------------------------------------------------------- preprocess
def resize_center_crop(frames: np.ndarray, size: int) -> torch.Tensor:
    """uint8 (T,H,W,3) -> float32 (T,3,size,size), shorter side resized then center-cropped."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    T, C, H, W = x.shape
    scale = size / min(H, W)
    nh, nw = max(int(round(H * scale)), size), max(int(round(W * scale)), size)
    x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    top, left = (nh - size) // 2, (nw - size) // 2
    return x[:, :, top : top + size, left : left + size].clamp(0, 1)


def patchify(
    frames: torch.Tensor,
    duplicate_frames: bool = True,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 (T,3,H,W) in [0,1] -> (pixel_values (S, patch_dim), grid_thw (3,)).

    duplicate_frames=True repeats every frame x2 so one temporal group (latent slot)
    corresponds to exactly one sampled frame (temporal_patch_size=2).
    """
    x = (frames - IMAGE_MEAN) / IMAGE_STD
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    if duplicate_frames:
        x = x.repeat_interleave(temporal_patch_size, dim=0)
    T = x.shape[0]
    if pad := -T % temporal_patch_size:
        x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)], dim=0)
    patches = x.unsqueeze(0)  # (1, T, C, H, W)
    batch_size = 1
    channel = patches.shape[2]
    grid_t = patches.shape[1] // temporal_patch_size
    resized_height, resized_width = patches.shape[-2], patches.shape[-1]
    grid_h, grid_w = resized_height // PATCH_SIZE, resized_width // PATCH_SIZE
    # --- begin verbatim HF layout ---
    patches = patches.view(
        batch_size,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
        grid_w // MERGE_SIZE,
        MERGE_SIZE,
        PATCH_SIZE,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        batch_size,
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * PATCH_SIZE * PATCH_SIZE,
    )
    # --- end verbatim HF layout ---
    grid = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
    return flatten_patches[0].contiguous(), grid
