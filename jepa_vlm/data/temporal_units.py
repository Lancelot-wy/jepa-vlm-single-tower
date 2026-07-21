"""Real-frame temporal-unit construction and diagnostics.

EXP-12 defines one Qwen temporal unit as two adjacent sampled images.  This
module is deliberately independent from the decoder and model so the ordering
contract can be tested with numbered synthetic frames.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class TemporalUnitDiagnostics:
    raw_frame_count: int
    unique_frame_count: int
    temporal_unit_count: int
    duplicate_adjacent_ratio: float
    effective_fps: float
    state_eligible: bool
    state_skipped_short: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def temporal_unit_frame_ids(frame_ids: np.ndarray, temporal_patch_size: int) -> np.ndarray:
    """Return `[num_units, temporal_patch_size]` frame IDs without duplication.

    EXP-12 rejects incomplete units instead of padding them with a repeated last
    frame.  Legacy padding remains available in :func:`video_io.patchify` for old
    configs only.
    """
    ids = np.asarray(frame_ids)
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    if len(ids) % temporal_patch_size:
        raise ValueError("real temporal units require an exact number of raw frames")
    return ids.reshape(len(ids) // temporal_patch_size, temporal_patch_size)


def build_temporal_units(frames: np.ndarray, temporal_patch_size: int) -> np.ndarray:
    """Group ordered raw frames into adjacent units without copying any frame."""
    if len(frames) % temporal_patch_size:
        raise ValueError("real temporal units require an exact number of raw frames")
    return np.asarray(frames).reshape(
        len(frames) // temporal_patch_size, temporal_patch_size, *frames.shape[1:]
    )


def state_pair_indices(num_temporal_units: int, horizon_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned source/target unit indices for the requested fixed horizon."""
    if horizon_units < 1:
        raise ValueError("horizon_units must be positive")
    count = num_temporal_units - horizon_units
    if count <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    source = np.arange(count, dtype=np.int64)
    return source, source + horizon_units


def temporal_diagnostics(
    sampled_frame_ids: np.ndarray,
    native_fps: float,
    temporal_patch_size: int,
    horizon_units: int,
    expected_fps: float | None = None,
) -> TemporalUnitDiagnostics:
    """Summarize whether sampled images can form a genuine future-state pair."""
    ids = np.asarray(sampled_frame_ids, dtype=np.int64)
    adjacent_duplicate = ids[1:] == ids[:-1]
    duplicate_ratio = float(adjacent_duplicate.mean()) if len(adjacent_duplicate) else 0.0
    units = len(ids) // temporal_patch_size
    exact = len(ids) == units * temporal_patch_size
    unique = int(np.unique(ids).size)
    unit_ids = ids[: units * temporal_patch_size].reshape(units, temporal_patch_size) if units else None
    # Every unit must contain different real images and source/target timestamps
    # must differ by the configured number of units.
    distinct_within = bool(unit_ids is not None and np.all(np.diff(unit_ids, axis=1) != 0))
    monotonic = bool(len(ids) < 2 or np.all(np.diff(ids) > 0))
    if len(ids) > 1 and native_fps > 0:
        duration = (float(ids[-1]) - float(ids[0])) / native_fps
        effective_fps = (len(ids) - 1) / duration if duration > 0 else 0.0
    else:
        effective_fps = 0.0
    fps_matches = bool(
        expected_fps is None
        or expected_fps <= 0
        or abs(effective_fps - expected_fps) / expected_fps <= 0.10
    )
    eligible = bool(
        exact and units > horizon_units and distinct_within and monotonic and fps_matches
    )
    return TemporalUnitDiagnostics(
        raw_frame_count=len(ids),
        unique_frame_count=unique,
        temporal_unit_count=units,
        duplicate_adjacent_ratio=duplicate_ratio,
        effective_fps=float(effective_fps),
        state_eligible=eligible,
        state_skipped_short=not eligible,
    )
