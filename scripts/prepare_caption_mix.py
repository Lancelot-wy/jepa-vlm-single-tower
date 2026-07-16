"""Build a deterministic, locally-resolved caption-as-QA manifest.

Caption sources are represented as normal Phase-B records so that both the CE
control and CE+MSE treatment see *exactly* the same videos/text.  No row is
emitted unless its local media file exists.  Sampling is reservoir based, so
the selected subset is uniform over valid records rather than directory order.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

from data_sources import (VideoResolver, iter_source_records, load_registry, optional_number,
                          qa_examples_for, select_sources, source_id_for)


def build_source(name: str, source: dict, seed: int, progress_every: int) -> tuple[list[dict], dict]:
    limit = int(source["max_samples"])
    minimum = int(source["min_samples"])
    rng = random.Random(f"{seed}:{name}")
    resolver = VideoResolver(source)
    reservoir: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stats = collections.Counter()
    started = time.monotonic()

    for record, provenance in iter_source_records(source):
        stats["records_seen"] += 1
        pairs = qa_examples_for(record, source)
        if not pairs:
            stats["missing_caption"] += 1
        else:
            video = resolver.resolve(record)
            if not video:
                stats["missing_video"] += 1
            else:
                source_id = source_id_for(record, source, fallback=os.path.splitext(os.path.basename(video))[0])
                start = optional_number(record, list(source.get("start_fields", [])))
                end = optional_number(record, list(source.get("end_fields", [])))
                for question, answer in pairs:
                    # Native LLaVA records can have two different QA turns per video;
                    # retain both while still suppressing exact repeated turns.
                    key = (source_id, video, question)
                    if key in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(key)
                    item = {
                        "video": video,
                        "question": question,
                        "answer": answer,
                        "source_dataset": name,
                        "source_category": str(record.get("category") or source.get("role") or "unknown"),
                        "source_id": source_id,
                        "provenance": provenance,
                    }
                    if start is not None and end is not None and end > start:
                        item["start"], item["end"] = start, end
                    stats["valid"] += 1
                    if len(reservoir) < limit:
                        reservoir.append(item)
                    else:
                        replacement = rng.randrange(stats["valid"])
                        if replacement < limit:
                            reservoir[replacement] = item
        if progress_every and stats["records_seen"] % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[prepare:{name}] scanned={stats['records_seen']} valid={stats['valid']} "
                f"selected={len(reservoir)} missing_video={stats['missing_video']} "
                f"rate={stats['records_seen'] / elapsed:.0f} records/s",
                flush=True,
            )

    stats["selected"] = len(reservoir)
    if len(reservoir) < minimum:
        raise RuntimeError(
            f"{name}: only {len(reservoir)} valid locally-resolved training examples; need at least {minimum}. "
            f"missing_video={stats['missing_video']}. Run audit_data_sources.py and fix the mapping first."
        )
    return reservoir, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--progress-every", type=int, default=10000,
                    help="emit bounded progress every N records (0 disables it)")
    args = ap.parse_args()

    registry = load_registry(args.registry)
    items: list[dict] = []
    reports: dict[str, dict] = {}
    for name, source in select_sources(registry, args.sources):
        selected, stats = build_source(name, source, args.seed, args.progress_every)
        items.extend(selected)
        reports[name] = stats
        print(f"{name}: {stats}")

    # Preserve multiple native QA turns from one source/video, but suppress an
    # exact physical video path appearing again under a lower-priority source.
    # This is intentionally conservative: it cannot identify re-encodes or
    # different mount paths that refer to the same source video.
    kept: list[dict] = []
    video_owner: dict[str, str] = {}
    cross_source_exact_path_duplicates = collections.Counter()
    for item in items:
        canonical_video = os.path.realpath(item["video"])
        owner = video_owner.get(canonical_video)
        if owner is not None and owner != item["source_dataset"]:
            cross_source_exact_path_duplicates[item["source_dataset"]] += 1
            continue
        video_owner.setdefault(canonical_video, item["source_dataset"])
        kept.append(item)
    items = kept
    selected_sources = dict(select_sources(registry, args.sources))
    final_counts = collections.Counter(item["source_dataset"] for item in items)
    for name, source in selected_sources.items():
        minimum = int(source["min_samples"])
        if final_counts[name] < minimum:
            raise RuntimeError(
                f"{name}: cross-source exact-path dedup left {final_counts[name]} rows; "
                f"need at least {minimum}. Resolve the overlap or lower the registry cap explicitly."
            )

    # Fix final ordering for reproducibility while retaining uniform per-source
    # reservoir sampling.  The train DataLoader shuffles independently per seed.
    items.sort(key=lambda item: (item["source_dataset"], item["source_id"], item["video"]))
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)

    report_path = args.report or args.out + ".report.json"
    report = {
        "manifest": os.path.abspath(args.out),
        "rows": len(items),
        "by_source": {name: sum(1 for item in items if item["source_dataset"] == name) for name in reports},
        "by_source_category": dict(collections.Counter(item["source_category"] for item in items)),
        "source_stats": reports,
        "cross_source_exact_path_duplicates_dropped": dict(cross_source_exact_path_duplicates),
        "caveat": "Exact real-path duplicates across sources are removed. Re-encoded, renamed, or separately-mounted copies are not detectable here; run benchmark ID/path filtering next.",
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(items)} rows -> {args.out}\nreport -> {report_path}")


if __name__ == "__main__":
    main()
