"""Build jepa-vlm manifests from a LOCAL LLaVA-Video-178K copy (Phase A getting-started).

LLaVA-Video ships per-subset jsonl under <root>/jsonl/final_<subset>_*_processed.jsonl
whose lines carry an absolute `video_path` to an already-extracted video. Phase A is
self-supervised (only `video` is needed), so we just dedup the video paths, drop any
that are missing on disk, and split into train/val manifests.

  python scripts/prepare_llava_video.py \
      --root /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K \
      --subsets 0_30_s_academic_v0_1 \
      --out-dir /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/llava_video \
      --max-videos 8000

Then set in your config (paths are absolute -> data_root=""):
  train.train_manifest=<out>/train.jsonl  train.val_manifest=<out>/val.jsonl
  train.data_root=""  train.min_flow=0.0
"""

import argparse
import glob
import json
import os
import random


def iter_video_paths(root: str, subsets: list[str]):
    jsonl_dir = os.path.join(root, "jsonl")
    files = sorted(glob.glob(os.path.join(jsonl_dir, "final_*_processed*.jsonl")))
    if subsets:
        files = [f for f in files if any(s in os.path.basename(f) for s in subsets)]
    if not files:
        raise FileNotFoundError(f"no matching jsonl in {jsonl_dir} for subsets={subsets}")
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vp = d.get("video_path") or d.get("video")
                if vp:
                    yield vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LLaVA-Video-178K root")
    ap.add_argument("--subsets", nargs="*", default=["0_30_s_academic_v0_1"],
                    help="subset name substrings to include (match jsonl filenames)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = no cap")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check-files", action="store_true",
                    help="skip os.path.exists filtering (faster, but may keep missing files)")
    args = ap.parse_args()

    seen: set[str] = set()
    n_missing = 0
    for vp in iter_video_paths(args.root, args.subsets):
        if vp in seen:
            continue
        if not args.no_check_files and not os.path.exists(vp):
            n_missing += 1
            continue
        seen.add(vp)
        if args.max_videos and len(seen) >= args.max_videos:
            break

    vids = sorted(seen)
    random.Random(args.seed).shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac)) if vids else 0
    val, train = vids[:n_val], vids[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w") as out:
            for vp in rows:
                out.write(json.dumps({"video": vp}) + "\n")
    print(f"videos: {len(vids)} (missing {n_missing}) -> train {len(train)} / val {len(val)}")
    print(f"wrote {args.out_dir}/train.jsonl , val.jsonl")
    print("set: train.data_root='' train.min_flow=0.0")


if __name__ == "__main__":
    main()
