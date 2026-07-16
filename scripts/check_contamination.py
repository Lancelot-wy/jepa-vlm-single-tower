"""训练 manifest vs 评测集的视频重叠检查（EXP-09 必做门槛）。

背景：LLaVA-Video-178K 成分含 PerceptionTest/Charades/ActivityNet/NExTQA——
前三者是 MVBench 部分子任务的源视频库，NExT-QA 是我们的评测集。任何重叠视频
必须从训练里剔除，否则该评测的数字作废。

比对键 = 视频路径/文件名、显式 video_id/clip_id/source_id 及评测侧的帧目录名。
保守策略：宁可误杀。这个检查能发现直接路径或 ID 重叠，不能证明没有改名、
重编码或裁剪后的同源视频；后者必须依赖训练来源白名单。

  python scripts/check_contamination.py --train jepa_data/exp09/qa_train.jsonl \
      --bench <MVBench jsonl> <TempCompass jsonl> <NextQA jsonl> \
      --clean-out jepa_data/exp09/qa_train_clean.jsonl
"""

import argparse
import json
import os
from urllib.parse import urlsplit


ID_FIELDS = ("video", "video_path", "source_id", "video_id", "videoID", "YoutubeID",
             "youtube_id", "clip_id", "id")


def _valid(key: str) -> bool:
    """Ignore generic one-character/numeric placeholders that cause false matches."""
    return len(key) >= 5 and any(ch.isalpha() for ch in key)


def keys_for_value(value) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    raw = value.strip()
    # Normalise URL-bearing records (common in web-video metadata) without
    # losing the final filename / YouTube-like identifier.
    parsed = urlsplit(raw)
    path = parsed.path if parsed.scheme else raw
    candidates = {raw.lower(), path.lower()}
    base = os.path.basename(path.rstrip("/"))
    if base:
        candidates.add(base.lower())
        candidates.add(os.path.splitext(base)[0].lower())
    return {candidate for candidate in candidates if _valid(candidate)}


def record_keys(d: dict, include_frame_parent: bool = False) -> set[str]:
    keys = set()
    for field in ID_FIELDS:
        keys |= keys_for_value(d.get(field))
    # Some evaluation records nest their source video description in `meta`.
    meta = d.get("meta") or {}
    if isinstance(meta, dict):
        for field in ID_FIELDS:
            keys |= keys_for_value(meta.get(field))
        images = meta.get("images_info") or []
        for image in images[:1]:  # all frames of one item have the same origin
            path = image.get("image") if isinstance(image, dict) else image
            keys |= keys_for_value(path)
            if include_frame_parent and path:
                keys |= keys_for_value(os.path.basename(os.path.dirname(str(path))))
    return keys


def bench_keys(path: str) -> set:
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # "视频" is used by the company's evaluation JSONL; other fields
            # cover source manifests emitted by the curated caption builder.
            keys |= keys_for_value(d.get("视频"))
            keys |= record_keys(d, include_frame_parent=True)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--clean-out", default="", help="写出剔除重叠后的干净 manifest")
    args = ap.parse_args()

    bkeys = set()
    for b in args.bench:
        k = bench_keys(b)
        print(f"{os.path.basename(b)}: {len(k)} 个评测视频键")
        bkeys |= k

    kept, dropped = [], []
    with open(args.train) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            matched = record_keys(d) & bkeys
            (dropped if matched else kept).append((d, matched))

    total = len(kept) + len(dropped)
    print(f"\n训练集 {total} 条，与评测重叠 {len(dropped)} 条 "
          f"({100 * len(dropped) / max(total, 1):.2f}%)")
    for d, matched in dropped[:10]:
        print("  重叠样例:", d.get("video"), "keys=", sorted(matched)[:3])
    if args.clean_out:
        with open(args.clean_out, "w") as out:
            for d, _ in kept:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"干净 manifest ({len(kept)} 条) -> {args.clean_out}")
    if dropped and not args.clean_out:
        raise SystemExit("存在重叠且未指定 --clean-out，拒绝默默通过。")


if __name__ == "__main__":
    main()
