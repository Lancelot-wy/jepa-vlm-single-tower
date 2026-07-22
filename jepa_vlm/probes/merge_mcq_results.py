"""Merge deterministic MCQ-evaluation shards into one paired-results document."""

from __future__ import annotations

import argparse
import json
import os
import statistics

from .mcq_utils import result_document


def merge_documents(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("no documents to merge")
    for field in ("task", "protocol", "scoring"):
        values = {document.get(field) for document in documents}
        if len(values) != 1:
            raise ValueError(f"shard {field} mismatch: {sorted(map(str, values))}")
    records = [record for document in documents for record in document.get("results", [])]
    indices = [int(record["idx"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("shards contain duplicate item indices")
    records.sort(key=lambda record: int(record["idx"]))
    skipped = sum(int(document.get("skipped", 0)) for document in documents)
    metadata = {
        "merged_shards": len(documents),
        "shards": [document.get("metadata", {}) for document in documents],
    }
    token_counts = [
        int(record["native_video_tokens"])
        for record in records if record.get("native_video_tokens") is not None
    ]
    if token_counts:
        metadata["native_video_tokens"] = {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
        }
    return result_document(
        task=documents[0]["task"],
        protocol=documents[0]["protocol"],
        scoring=documents[0]["scoring"],
        records=records,
        skipped=skipped,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    documents = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            documents.append(json.load(handle))
    merged = merge_documents(documents)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)
    print(json.dumps({
        "output": args.output,
        "task": merged["task"],
        "protocol": merged["protocol"],
        "correct": merged["correct"],
        "total": merged["total"],
        "acc": merged["acc"],
        "skipped": merged["skipped"],
    }, indent=2))


if __name__ == "__main__":
    main()
