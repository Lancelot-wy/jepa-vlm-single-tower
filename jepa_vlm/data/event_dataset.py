"""Audited event-transition dataset for the disabled-by-default EXP-12 B sweep."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_io import decode_frames, patchify, resize_center_crop


REQUIRED_EVENT_FIELDS = (
    "video_path", "video_id", "source_event_id", "target_event_id",
    "source_start", "source_end", "target_start", "target_end", "direction",
    "event_caption", "source_dataset", "source_id",
)


@dataclass(frozen=True)
class EventRecord:
    video_path: str
    video_id: str
    source_event_id: str
    target_event_id: str
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    direction: str
    event_caption: str
    source_dataset: str
    source_id: str


def parse_event_record(value: dict) -> EventRecord:
    missing = [key for key in REQUIRED_EVENT_FIELDS if value.get(key) in (None, "")]
    if missing:
        raise ValueError(f"event record is missing fields: {missing}")
    record = EventRecord(**{key: value[key] for key in REQUIRED_EVENT_FIELDS})
    record = EventRecord(
        **{
            **record.__dict__,
            "source_start": float(record.source_start),
            "source_end": float(record.source_end),
            "target_start": float(record.target_start),
            "target_end": float(record.target_end),
        }
    )
    if record.direction not in ("previous", "next"):
        raise ValueError("event direction must be previous or next")
    if not (0 <= record.source_start < record.source_end):
        raise ValueError("invalid source event boundary")
    if not (0 <= record.target_start < record.target_end):
        raise ValueError("invalid target event boundary")
    if record.source_event_id == record.target_event_id:
        raise ValueError("source and target event IDs must differ")
    if record.direction == "next" and record.target_start < record.source_start:
        raise ValueError("next target precedes its source")
    if record.direction == "previous" and record.target_start > record.source_start:
        raise ValueError("previous target follows its source")
    return record


def validate_event_duration(record: EventRecord, duration_seconds: float) -> None:
    if not duration_seconds > 0:
        raise ValueError("video duration must be positive")
    if max(record.source_end, record.target_end) > duration_seconds + 1e-3:
        raise ValueError(
            f"event boundary exceeds video duration {duration_seconds:.3f}s: {record}"
        )


def _validate_adjacent_events(records: list[EventRecord]) -> None:
    """Verify pair direction and adjacency against all known events per video."""
    by_video: dict[str, list[EventRecord]] = {}
    for record in records:
        by_video.setdefault(record.video_id, []).append(record)
    for video_id, rows in by_video.items():
        paths = {row.video_path for row in rows}
        if len(paths) != 1:
            raise ValueError(f"video_id {video_id} maps to multiple video paths")
        boundaries: dict[str, tuple[float, float]] = {}
        for row in rows:
            for event_id, boundary in (
                (row.source_event_id, (row.source_start, row.source_end)),
                (row.target_event_id, (row.target_start, row.target_end)),
            ):
                old = boundaries.setdefault(event_id, boundary)
                if old != boundary:
                    raise ValueError(
                        f"event {video_id}/{event_id} has inconsistent boundaries"
                    )
        ordered = sorted(boundaries, key=lambda event_id: boundaries[event_id][0])
        positions = {event_id: index for index, event_id in enumerate(ordered)}
        for row in rows:
            delta = positions[row.target_event_id] - positions[row.source_event_id]
            expected = 1 if row.direction == "next" else -1
            if delta != expected:
                raise ValueError(
                    f"{video_id}: target {row.target_event_id} is not the adjacent "
                    f"{row.direction} event of {row.source_event_id}"
                )


def _split_name(video_id: str, seed: int, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{video_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if value < val_fraction else "train"


def load_event_records(
    path: str,
    *,
    split: str | None = None,
    seed: int = 0,
    val_fraction: float = 0.05,
) -> list[EventRecord]:
    if split not in (None, "train", "val"):
        raise ValueError("split must be train, val, or None")
    all_records = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = parse_event_record(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid event row {line_number}: {exc}") from exc
            all_records.append(record)
    _validate_adjacent_events(all_records)
    records = [
        record for record in all_records
        if split is None or _split_name(record.video_id, seed, val_fraction) == split
    ]
    if not records:
        raise ValueError(f"event manifest has no records for split={split}: {path}")
    return records


def assert_video_split_isolation(train: list[EventRecord], val: list[EventRecord]) -> None:
    overlap = {row.video_id for row in train} & {row.video_id for row in val}
    if overlap:
        raise ValueError(f"event train/val video_id overlap: {sorted(overlap)[:5]}")


class EventVideoDataset(Dataset):
    """Decode source, true target, and same-video wrong-event target clips."""

    def __init__(
        self,
        manifest: str,
        *,
        split: str,
        raw_num_frames: int,
        sample_fps: float,
        frame_sampling: str,
        frame_size: int,
        temporal_patch_size: int,
        seed: int,
        inner_min: float,
        inner_max: float,
        direction_mode: str = "bidirectional",
    ):
        self.records = load_event_records(manifest, split=split, seed=seed)
        if direction_mode not in ("forward", "backward", "bidirectional"):
            raise ValueError("bad event direction mode")
        if direction_mode != "bidirectional":
            wanted = "next" if direction_mode == "forward" else "previous"
            self.records = [record for record in self.records if record.direction == wanted]
        if not self.records:
            raise ValueError(f"event manifest has no {direction_mode} rows for split={split}")
        self.raw_num_frames = raw_num_frames
        self.sample_fps = sample_fps
        self.frame_sampling = frame_sampling
        self.frame_size = frame_size
        self.temporal_patch_size = temporal_patch_size
        self.seed = seed
        self.inner_min = inner_min
        self.inner_max = inner_max
        self.by_video: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            self.by_video.setdefault(record.video_id, []).append(index)
        durations = {}
        for record in self.records:
            if record.video_path not in durations:
                if not os.path.isfile(record.video_path):
                    raise ValueError(f"event video does not exist: {record.video_path}")
                import av

                with av.open(record.video_path) as container:
                    stream = container.streams.video[0]
                    if stream.duration is not None:
                        duration = float(stream.duration * stream.time_base)
                    elif container.duration is not None:
                        duration = float(container.duration / av.time_base)
                    else:
                        raise ValueError(f"cannot determine video duration: {record.video_path}")
                durations[record.video_path] = duration
            validate_event_duration(record, durations[record.video_path])
        for index, record in enumerate(self.records):
            if not self._negative_candidates(record):
                raise ValueError(
                    f"video {record.video_id} has no same-video wrong-event target for row {index}"
                )

    def __len__(self):
        return len(self.records)

    def _negative_candidates(self, record: EventRecord) -> list[EventRecord]:
        return [
            self.records[index] for index in self.by_video.get(record.video_id, [])
            if self.records[index].target_event_id != record.target_event_id
            and (
                self.records[index].target_start,
                self.records[index].target_end,
            ) != (record.target_start, record.target_end)
        ]

    def _decode(self, record: EventRecord, target: bool) -> tuple[torch.Tensor, torch.Tensor]:
        start = record.target_start if target else record.source_start
        end = record.target_end if target else record.source_end
        frames = decode_frames(
            record.video_path, self.raw_num_frames, self.sample_fps, self.frame_sampling,
            start=start, end=end, random_offset=False,
        )
        return patchify(
            resize_center_crop(frames, self.frame_size), duplicate_frames=False,
            temporal_patch_size=self.temporal_patch_size,
        )

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rng = np.random.default_rng(self.seed * 100003 + index)
        negatives = self._negative_candidates(record)
        negative = negatives[int(rng.integers(0, len(negatives)))]
        source_pixels, source_grid = self._decode(record, target=False)
        target_pixels, target_grid = self._decode(record, target=True)
        negative_pixels, negative_grid = self._decode(negative, target=True)
        return {
            "source_pixel_values": source_pixels,
            "source_grid_thw": source_grid,
            "target_pixel_values": target_pixels,
            "target_grid_thw": target_grid,
            "negative_pixel_values": negative_pixels,
            "negative_grid_thw": negative_grid,
            "direction": 1 if record.direction == "next" else 0,
            "condition": f"{record.direction}: {record.event_caption}",
            "target_inner_fraction": float(rng.uniform(self.inner_min, self.inner_max)),
            "source_inner_fraction": 0.8 if record.direction == "next" else 0.2,
            "video_id": record.video_id,
            "target_event_id": record.target_event_id,
            "negative_event_id": negative.target_event_id,
        }


class EventCollator:
    def __init__(self, tokenizer, max_condition_tokens: int = 128):
        self.tokenizer = tokenizer
        self.max_condition_tokens = max_condition_tokens

    def __call__(self, batch: list[dict]) -> dict:
        encoded = [
            self.tokenizer.encode(row["condition"], add_special_tokens=False)[
                : self.max_condition_tokens
            ]
            for row in batch
        ]
        length = max(max(map(len, encoded), default=0), 1)
        pad = self.tokenizer.pad_token_id or 0
        condition_ids = torch.full((len(batch), length), pad, dtype=torch.long)
        condition_mask = torch.zeros(len(batch), length, dtype=torch.long)
        for index, values in enumerate(encoded):
            if values:
                condition_ids[index, : len(values)] = torch.tensor(values)
                condition_mask[index, : len(values)] = 1
        return {
            key: torch.stack([row[key] for row in batch])
            for key in ("source_pixel_values", "target_pixel_values", "negative_pixel_values")
        } | {
            "source_grid_thw": batch[0]["source_grid_thw"],
            "target_grid_thw": batch[0]["target_grid_thw"],
            "negative_grid_thw": batch[0]["negative_grid_thw"],
            "condition_input_ids": condition_ids,
            "condition_attention_mask": condition_mask,
            "direction": torch.tensor([row["direction"] for row in batch], dtype=torch.long),
            "source_inner_fraction": torch.tensor(
                [row["source_inner_fraction"] for row in batch], dtype=torch.float32
            ),
            "target_inner_fraction": torch.tensor(
                [row["target_inner_fraction"] for row in batch], dtype=torch.float32
            ),
            "video_id": [row["video_id"] for row in batch],
            "target_event_id": [row["target_event_id"] for row in batch],
            "negative_event_id": [row["negative_event_id"] for row in batch],
        }
