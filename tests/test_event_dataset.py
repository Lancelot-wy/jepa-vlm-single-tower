import json

import pytest

from jepa_vlm.data.event_dataset import (
    assert_video_split_isolation,
    load_event_records,
    parse_event_record,
    validate_event_duration,
)


def row(video_id, source_id, target_id, direction="next", offset=0.0):
    if direction == "next":
        source_start, target_start = offset, offset + 2
    else:
        source_start, target_start = offset + 2, offset
    return {
        "video_path": "/tmp/video.mp4", "video_id": video_id,
        "source_event_id": source_id, "target_event_id": target_id,
        "source_start": source_start, "source_end": source_start + 1,
        "target_start": target_start, "target_end": target_start + 1,
        "direction": direction, "event_caption": "a transition",
        "source_dataset": "unit", "source_id": f"{video_id}-{source_id}-{target_id}",
    }


def test_next_previous_and_invalid_boundaries():
    parsed = parse_event_record(row("v", "a", "b", "next"))
    assert parsed.direction == "next"
    validate_event_duration(parsed, 4.0)
    with pytest.raises(ValueError):
        validate_event_duration(parsed, 2.5)
    assert parse_event_record(row("v", "b", "a", "previous")).direction == "previous"
    bad = row("v", "a", "b")
    bad["target_end"] = bad["target_start"]
    with pytest.raises(ValueError):
        parse_event_record(bad)


def test_video_id_split_isolation(tmp_path):
    path = tmp_path / "events.jsonl"
    with path.open("w") as handle:
        for index in range(100):
            handle.write(json.dumps(row(f"v{index}", "a", "b")) + "\n")
    train = load_event_records(str(path), split="train", seed=3, val_fraction=0.2)
    val = load_event_records(str(path), split="val", seed=3, val_fraction=0.2)
    assert_video_split_isolation(train, val)


def test_non_adjacent_event_pair_is_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        row("v", "a", "b", offset=0.0),
        row("v", "b", "c", offset=2.0),
        row("v", "a", "c", offset=0.0),
    ]
    # Make the third target agree with event c's boundary while skipping b.
    rows[2]["target_start"], rows[2]["target_end"] = 4.0, 5.0
    with path.open("w") as handle:
        for value in rows:
            handle.write(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="not the adjacent"):
        load_event_records(str(path))
