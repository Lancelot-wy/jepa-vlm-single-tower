"""NExT-QA 官方 train split -> MCQ 训练 manifest（EXP-09 数据扩容）。

只允许处理 train split（val/test 是评测资产，谁进训练谁污染）。官方 meta 为 CSV：
  video, question, answer(0-4 的索引), type, a0..a4
视频定位：--video-root 下的 <video>.mp4，或经 --vid-map（map_vid_vidorID.json，
id -> "dir/sub" 相对路径）解析。字段名不符时打印实际表头并退出，不猜。

  python scripts/prepare_nextqa.py --csv <meta>/train.csv \
      --video-root <NExTQA 视频目录> [--vid-map <meta>/map_vid_vidorID.json] \
      --out jepa_data/exp09/nextqa_train.jsonl

输出行: {"video": 绝对路径, "question": 题干+Options块, "answer": "(B) xxx", "source": "nextqa"}
"""

import argparse
import csv
import json
import os

LETTERS = "ABCDE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NExT-QA train.csv（禁止传 val/test）")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--vid-map", default="", help="map_vid_vidorID.json（可选）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exts", default=".mp4,.avi,.webm")
    args = ap.parse_args()

    base = os.path.basename(args.csv).lower()
    if "train" not in base:
        raise SystemExit(f"拒绝执行：{base} 看起来不是 train split。val/test 是评测资产。")

    vid_map = {}
    if args.vid_map:
        with open(args.vid_map) as f:
            vid_map = json.load(f)

    exts = args.exts.split(",")

    def resolve(vid: str):
        cands = [vid_map.get(str(vid), str(vid)), str(vid)]
        for c in cands:
            for e in exts:
                p = os.path.join(args.video_root, c if c.endswith(tuple(exts)) else c + e)
                if os.path.exists(p):
                    return p
        return None

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV 为空")
    cols = set(rows[0].keys())
    need = {"video", "question", "answer"}
    opt_cols = [c for c in ("a0", "a1", "a2", "a3", "a4") if c in cols]
    if not need.issubset(cols) or len(opt_cols) < 2:
        raise SystemExit(f"字段不符，实际表头: {sorted(cols)}\n"
                         f"需要 video/question/answer + a0..a4，请核对后改本脚本的列名映射。")

    n_ok = n_miss = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out:
        for r in rows:
            path = resolve(r["video"])
            if path is None:
                n_miss += 1
                continue
            opts = [r[c].strip() for c in opt_cols if r[c].strip()]
            try:
                ai = int(r["answer"])
            except ValueError:
                n_miss += 1
                continue
            if ai >= len(opts):
                n_miss += 1
                continue
            block = "\n".join(f"({LETTERS[k]}) {o}" for k, o in enumerate(opts))
            q = (r["question"].strip().rstrip("?") + "?\nOptions:\n" + block +
                 "\nAnswer with the option's letter.")
            a = f"({LETTERS[ai]}) {opts[ai]}"
            out.write(json.dumps({"video": path, "question": q, "answer": a,
                                  "source": "nextqa", "qtype": r.get("type", "")},
                                 ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"written {n_ok} (missing/bad {n_miss}) -> {args.out}")
    with open(args.out) as f:
        print("sample:", f.readline()[:300])


if __name__ == "__main__":
    main()
