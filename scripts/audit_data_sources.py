"""Check mounted video-text sources before they enter a training manifest.

This verifies the things a path list cannot: readable metadata, usable caption
fields, and local decodable-file *paths* for representative records.  It does
not assert a source has zero re-encoded benchmark duplicates; that claim would
need source-level provenance beyond a file-path comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from data_sources import (VideoResolver, caption_for, iter_source_records, load_registry,
                          metadata_files, select_sources, source_id_for)


def audit_one(name: str, source: dict, sample: int) -> dict:
    metadata = metadata_files(source)
    result = {
        "name": name,
        "role": source.get("role", ""),
        "enabled": bool(source.get("enabled", False)),
        "required": bool(source.get("required", False)),
        "metadata_entries": source.get("metadata", []),
        "metadata_files": len(metadata),
        "video_root": source.get("video_root", ""),
        "video_root_accessible": bool(source.get("video_root")) and os.path.isdir(source["video_root"]),
        "sampled_records": 0,
        "caption_present": 0,
        "video_resolved": 0,
        "examples": [],
    }
    resolver = VideoResolver(source)
    for record, provenance in iter_source_records(source):
        result["sampled_records"] += 1
        caption = caption_for(record, source)
        video = resolver.resolve(record) if caption else ""
        result["caption_present"] += bool(caption)
        result["video_resolved"] += bool(video)
        if len(result["examples"]) < 3:
            result["examples"].append({
                "source_id": source_id_for(record, source),
                "caption_chars": len(caption),
                "video": video,
                "provenance": provenance,
            })
        if result["sampled_records"] >= sample:
            break
    enough = min(3, sample)
    result["ready"] = bool(metadata) and result["video_root_accessible"] and \
        result["caption_present"] >= enough and result["video_resolved"] >= enough
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--sample", type=int, default=12, help="records inspected per source")
    ap.add_argument("--out", default="", help="optional JSON audit report")
    ap.add_argument("--strict", action="store_true", help="exit nonzero unless every selected source is ready")
    args = ap.parse_args()
    if args.sample < 3:
        raise SystemExit("--sample must be at least 3")

    registry = load_registry(args.registry)
    results = [audit_one(name, source, args.sample) for name, source in select_sources(registry, args.sources)]
    for result in results:
        print(
            f"{result['name']}: ready={result['ready']} metadata_files={result['metadata_files']} "
            f"video_root={result['video_root_accessible']} captions={result['caption_present']}/"
            f"{result['sampled_records']} resolved={result['video_resolved']}/{result['sampled_records']}"
        )
        for example in result["examples"]:
            print(f"  example id={example['source_id']!r} video={example['video'] or '<unresolved>'}")
    report = {"registry": os.path.abspath(args.registry), "sources": results,
              "caveat": "Path/ID audit; does not detect re-encoded or renamed duplicate videos."}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"audit report -> {args.out}")
    if args.strict and not all(result["ready"] for result in results):
        raise SystemExit("source audit failed: do not start training until selected sources resolve local videos")


if __name__ == "__main__":
    main()
