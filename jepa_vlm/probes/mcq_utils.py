"""Shared parsing and result helpers for multiple-choice video evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable


# Matches ``(A) foo``, ``A. foo``, ``A) foo`` and ``A: foo``.
_OPTION_RE = re.compile(r"^\s*\(?([A-H])[\).:.]\s*(.+?)\s*$", re.IGNORECASE)


def parse_options(question: str) -> list[tuple[str, str]]:
    """Return ``(letter, full option line)`` pairs from one benchmark question."""
    options = []
    for line in question.splitlines():
        match = _OPTION_RE.match(line)
        if match:
            options.append((match.group(1).upper(), line.strip()))
    return options


def target_letter(target: str) -> str | None:
    """Parse a benchmark target into an uppercase option letter."""
    match = _OPTION_RE.match(target.strip())
    if match:
        return match.group(1).upper()
    match = re.match(r"\s*\(?([A-H])\b", target.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def generated_letter(text: str, valid_letters: Iterable[str]) -> str | None:
    """Extract the first standalone valid letter from a generated answer."""
    text = text.strip()
    valid = {letter.upper() for letter in valid_letters}
    for pattern in (
        r"^\s*\(?([A-H])(?:\)|[\.:,]|$)",
        r"\b(?:option|answer)\s*(?:is|:)?\s*\(?([A-H])\b",
        r"\b(?:choose|select)\s*\(?([A-H])\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()
    return None


def category_summary(records: list[dict]) -> dict[str, dict[str, float | int]]:
    """Aggregate correct/total/accuracy by the benchmark ``sub_type`` field."""
    grouped: dict[str, list[int]] = {}
    for record in records:
        key = str(record.get("sub_type") or "unknown").strip() or "unknown"
        grouped.setdefault(key, [0, 0])
        grouped[key][0] += int(record.get("ok", 0))
        grouped[key][1] += 1
    return {
        key: {"correct": correct, "total": total, "acc": correct / max(total, 1)}
        for key, (correct, total) in sorted(grouped.items())
    }


def result_document(
    *,
    task: str,
    protocol: str,
    scoring: str,
    records: list[dict],
    skipped: int,
    metadata: dict | None = None,
) -> dict:
    """Build the common, shard-mergeable evaluator JSON schema."""
    correct = sum(int(record.get("ok", 0)) for record in records)
    total = len(records)
    return {
        "schema_version": 2,
        "task": task,
        "protocol": protocol,
        "scoring": scoring,
        "acc": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "skipped": int(skipped),
        "categories": category_summary(records),
        "metadata": metadata or {},
        "results": records,
    }
