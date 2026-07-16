"""训练 manifest vs 评测集的视频重叠检查（EXP-09 必做门槛）。

背景：LLaVA-Video-178K 成分含 PerceptionTest/Charades/ActivityNet/NExTQA——
前三者是 MVBench 部分子任务的源视频库，NExT-QA 是我们的评测集。任何重叠视频
必须从训练里剔除，否则该评测的数字作废。

比对键 = 视频文件 basename 去扩展名（评测侧从 meta.images_info 帧路径或 "视频"
字段推导；帧路径取其父目录名）。保守策略：宁可误杀。

  python scripts/check_contamination.py --train jepa_data/exp09/qa_train.jsonl \
      --bench <MVBench jsonl> <TempCompass jsonl> <NextQA jsonl> \
      --clean-out jepa_data/exp09/qa_train_clean.jsonl
"""

import argparse
import json
import os


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(str(p).rstrip("/")))[0].lower()


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
            v = d.get("视频") or d.get("video") or d.get("video_path")
            if v:
                keys.add(stem(v))
            info = (d.get("meta") or {}).get("images_info") or []
            for im in info[:1]:  # 每题帧同源，取一张即可
                p = im.get("image") if isinstance(im, dict) else im
                if p:
                    keys.add(stem(os.path.dirname(p)))  # 帧目录名通常=视频 id
                    keys.add(stem(p))
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
            (dropped if stem(d["video"]) in bkeys else kept).append(d)

    total = len(kept) + len(dropped)
    print(f"\n训练集 {total} 条，与评测重叠 {len(dropped)} 条 "
          f"({100 * len(dropped) / max(total, 1):.2f}%)")
    for d in dropped[:10]:
        print("  重叠样例:", d["video"])
    if args.clean_out:
        with open(args.clean_out, "w") as out:
            for d in kept:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"干净 manifest ({len(kept)} 条) -> {args.clean_out}")
    if dropped and not args.clean_out:
        raise SystemExit("存在重叠且未指定 --clean-out，拒绝默默通过。")


if __name__ == "__main__":
    main()
