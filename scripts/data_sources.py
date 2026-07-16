"""Shared, strict utilities for local video-text source registries.

The project receives mounted datasets whose metadata layouts differ.  These
helpers intentionally require a real local video file before emitting a row:
an accessible metadata directory alone is not enough to call a source usable.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections.abc import Iterator
from typing import Any

import yaml


MEDIA_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi", ".mov")
METADATA_EXTENSIONS = (".jsonl", ".json", ".csv")


def load_registry(path: str) -> dict[str, Any]:
    with open(path) as f:
        registry = yaml.safe_load(f)
    if not isinstance(registry, dict) or not isinstance(registry.get("sources"), dict):
        raise ValueError(f"registry {path} must contain a top-level 'sources' mapping")
    return registry


def select_sources(registry: dict[str, Any], names: list[str] | None) -> list[tuple[str, dict[str, Any]]]:
    sources = registry["sources"]
    if names:
        unknown = [name for name in names if name not in sources]
        if unknown:
            raise ValueError(f"unknown source(s): {', '.join(unknown)}")
        chosen = names
    else:
        chosen = [name for name, cfg in sources.items() if cfg.get("enabled", False)]
    if not chosen:
        raise ValueError("no selected sources; enable one in the registry or pass --sources")
    return [(name, sources[name]) for name in chosen]


def _metadata_files(entry: str) -> list[str]:
    if os.path.isdir(entry):
        found: list[str] = []
        for root, _, files in os.walk(entry):
            found.extend(
                os.path.join(root, name)
                for name in files
                if name.lower().endswith(METADATA_EXTENSIONS)
            )
        return sorted(found)
    return sorted(path for path in glob.glob(entry, recursive=True) if os.path.isfile(path))


def metadata_files(source: dict[str, Any]) -> list[str]:
    # LLaVA records live in several supported layouts and are enumerated by
    # prepare_llava_video.iter_records().  Do not recursively stat every media
    # file merely to make the audit's metadata-count field nonzero.
    if source.get("reader") == "llava":
        root = str(source.get("root") or source.get("video_root") or "")
        return [root] if os.path.isdir(root) else []
    files: list[str] = []
    for entry in source.get("metadata", []):
        files.extend(_metadata_files(str(entry)))
    return sorted(set(files))


def get_field(record: dict[str, Any], dotted_name: str) -> Any:
    """Read a dotted field while tolerating media lists in processed records.

    The unified caption/grounding export may represent a video as ``video``,
    as a nested media object, or as a list of media objects.  Returning a list
    for an un-indexed list path lets :class:`VideoResolver` try every explicit
    candidate without scanning a media directory.
    """
    keys = dotted_name.split(".")

    def visit(value: Any, remaining: list[str]) -> Any:
        if not remaining:
            return value
        key, rest = remaining[0], remaining[1:]
        if isinstance(value, dict):
            return visit(value.get(key), rest)
        if isinstance(value, (list, tuple)):
            if key.isdigit():
                index = int(key)
                return visit(value[index], rest) if 0 <= index < len(value) else None
            found = [visit(item, remaining) for item in value]
            return [item for item in found if item is not None]
        return None

    return visit(record, keys)


def first_text(record: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = get_field(record, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def first_value(record: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = get_field(record, field)
        if value not in (None, ""):
            return value
    return None


def clean_text(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-") + "."


def caption_for(record: dict[str, Any], source: dict[str, Any]) -> str:
    """Return a bounded answer, preserving Vript camera context when present."""
    main = first_text(record, list(source.get("caption_fields", [])))
    if not main:
        return ""
    context = []
    for field in source.get("caption_context_fields", []):
        value = first_text(record, [field])
        if value and value.lower() not in main.lower():
            label = field.rsplit(".", 1)[-1].replace("_", " ")
            context.append(f"{label}: {value}")
    answer = ". ".join(context + [main]) if context else main
    return clean_text(answer, int(source.get("max_answer_chars", 600)))


def qa_examples_for(record: dict[str, Any], source: dict[str, Any]) -> list[tuple[str, str]]:
    """Return one or more trainable QA pairs for a source record.

    Most sources are caption datasets and therefore become one fixed
    caption-as-answer pair.  LLaVA-Video is already an instruction dataset, so
    retain its native human/assistant pairs rather than discarding its QA form.
    """
    if source.get("reader") in ("llava", "conversation"):
        allowed_categories = {str(value).lower() for value in source.get("allowed_categories", [])}
        category = str(record.get("category", "")).lower()
        if allowed_categories and category not in allowed_categories:
            return []
        from prepare_llava_video import extract_qa

        limit = int(source.get("qa_per_video", 2))
        pairs = []
        for question, answer in extract_qa(record, max_pairs=limit):
            question = clean_text(question, int(source.get("max_question_chars", 720)))
            answer = clean_text(answer, int(source.get("max_answer_chars", 720)))
            if question and answer:
                pairs.append((question, answer))
        return pairs
    answer = caption_for(record, source)
    question = str(source.get("question", "")).strip()
    return [(question, answer)] if question and answer else []


def source_id_for(record: dict[str, Any], source: dict[str, Any], fallback: str = "") -> str:
    value = first_value(record, list(source.get("id_fields", [])))
    if value not in (None, ""):
        return str(value)
    return fallback


def _records_from_json(doc: Any, source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if isinstance(doc, list):
        for record in doc:
            if isinstance(record, dict):
                yield record
        return
    if not isinstance(doc, dict):
        return

    # Vript's released annotation is a top-level {meta, data} object with
    # clip IDs as keys below data.  Retain that key as a fallback identifier.
    data = doc.get("data")
    if isinstance(data, dict):
        for key, record in data.items():
            if isinstance(record, dict):
                record = dict(record)
                record.setdefault("clip_id", key)
                yield record
        return
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                yield record
        return

    for list_key in ("annotations", "items", "records", "samples"):
        records = doc.get(list_key)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    yield record
            return

    # A one-record JSON or a mapping keyed by clip ID.  The processed unified
    # export is conversation-first, so a grounding record can intentionally
    # have no legacy ``caption`` field at all.
    if source.get("reader") == "conversation" and isinstance(doc.get("conversations"), list):
        yield doc
        return
    if any(get_field(doc, field) not in (None, "") for field in source.get("caption_fields", [])):
        yield doc
        return
    for key, record in doc.items():
        if isinstance(record, dict):
            record = dict(record)
            record.setdefault("id", key)
            yield record


def _iter_jsonl_records(path: str) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _is_json_lines_file(path: str) -> bool:
    """Detect JSONL files that carry a misleading ``.json`` extension.

    A pretty JSON object starts with a partial line such as ``{`` and fails the
    single-line parse.  Two fully parseable, non-empty JSON object lines are a
    reliable enough signature for the sharded processed exports, and let us
    stream them rather than materializing a large shard with ``json.load``.
    """
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                return False
            if not isinstance(value, dict):
                return False
            records.append(value)
            if len(records) == 2:
                return True
    return False


def iter_source_records(source: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield (record, metadata_file) without inventing missing fields."""
    if source.get("reader") == "llava":
        # Reuse the established LLaVA resolver. It understands both the flat
        # jsonl/ and per-subset directory layouts and repairs stale absolute
        # paths embedded by another machine.
        from prepare_llava_video import iter_records

        root = str(source.get("root") or source.get("video_root") or "")
        if not root:
            raise ValueError("LLaVA source needs root or video_root")
        for video, record, source_jsonl in iter_records(
            root, [], list(source.get("exclude_patterns", []))
        ):
            record = dict(record)
            record["video_path"] = video
            record.setdefault("source_jsonl", source_jsonl)
            yield record, source_jsonl
        return
    for path in metadata_files(source):
        suffix = os.path.splitext(path)[1].lower()
        try:
            if suffix == ".jsonl":
                yield from ((record, path) for record in _iter_jsonl_records(path))
            elif suffix == ".csv":
                with open(path, newline="") as f:
                    for record in csv.DictReader(f):
                        yield dict(record), path
            elif suffix == ".json":
                if _is_json_lines_file(path):
                    yield from ((record, path) for record in _iter_jsonl_records(path))
                else:
                    with open(path) as f:
                        doc = json.load(f)
                    yield from ((record, path) for record in _records_from_json(doc, source))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"[data-source] skip unreadable metadata {path}: {exc}")


def _flat_record(record: dict[str, Any]) -> dict[str, str]:
    flat: dict[str, str] = {}

    def visit(value: Any, prefix: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif value not in (None, ""):
            flat[prefix] = str(value)
            flat.setdefault(prefix.rsplit(".", 1)[-1], str(value))

    visit(record)
    return flat


class VideoResolver:
    """Resolve explicit paths first; basename indexing is opt-in and bounded by source."""

    def __init__(self, source: dict[str, Any]):
        self.source = source
        self.root = str(source.get("video_root") or "")
        self._index: dict[str, str] | None = None

    def _candidate_values(self, record: dict[str, Any]) -> list[str]:
        def strings_from(value: Any) -> list[str]:
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            if isinstance(value, (list, tuple)):
                return [text for child in value for text in strings_from(child)]
            if isinstance(value, dict):
                media_keys = ("path", "video", "video_path", "filepath", "file_path",
                              "local_path", "uri", "url")
                return [text for key in media_keys for text in strings_from(value.get(key))]
            return []

        values: list[str] = []
        for field in self.source.get("video_fields", []):
            value = get_field(record, field)
            values.extend(strings_from(value))
        flat = _flat_record(record)
        for template in self.source.get("path_templates", []):
            try:
                value = str(template).format_map(flat)
            except (KeyError, ValueError):
                continue
            if value and "{" not in value:
                values.append(value)
        for field in self.source.get("id_fields", []):
            value = get_field(record, field)
            if value not in (None, ""):
                values.append(str(value))
        seen = set()
        expanded = []
        for value in values:
            candidates = [value]
            if not os.path.splitext(value)[1]:
                candidates.extend(value + ext for ext in MEDIA_EXTENSIONS)
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
        return expanded

    def _direct_paths(self, value: str) -> Iterator[str]:
        if os.path.isabs(value):
            yield value
        if self.root:
            yield os.path.join(self.root, value)
            yield os.path.join(self.root, os.path.basename(value))

    def _build_index(self):
        if self._index is not None:
            return
        if not self.root or not os.path.isdir(self.root):
            self._index = {}
            return
        index: dict[str, str] = {}
        count = 0
        print(f"[data-source] indexing media basenames under {self.root} (one-time)")
        for directory, _, names in os.walk(self.root):
            for name in names:
                if name.lower().endswith(MEDIA_EXTENSIONS):
                    index.setdefault(name.lower(), os.path.join(directory, name))
                    count += 1
        self._index = index
        print(f"[data-source] indexed {count} media files")

    def resolve(self, record: dict[str, Any]) -> str:
        candidates = self._candidate_values(record)
        for value in candidates:
            for path in self._direct_paths(value):
                if os.path.isfile(path):
                    return os.path.abspath(path)
        if self.source.get("index_fallback", False):
            self._build_index()
            assert self._index is not None
            for value in candidates:
                matched = self._index.get(os.path.basename(value).lower())
                if matched:
                    return matched
        return ""


def optional_number(record: dict[str, Any], fields: list[str]) -> float | None:
    value = first_value(record, fields)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
