"""Build a Phase-A manifest from Ego4D clips (or any directory of long videos, e.g.
EPIC-KITCHENS): scans for videos and windows them into fixed-length segments.

No labels are produced - Ego4D here is self-supervised pretraining data only.

  python scripts/prepare_ego4d.py --video-dir /data/ego4d/clips --out data/ego4d/train.jsonl \
      --window 16 --stride 12 --max-clips 200000
"""

import argparse
import json
import os
import random


def video_duration(path: str) -> float | None:
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return c.duration / av.time_base
            s = c.streams.video[0]
            if s.duration and s.time_base:
                return float(s.duration * s.time_base)
    except Exception as e:
        print(f"skip {path}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=16.0, help="segment length (s)")
    ap.add_argument("--stride", type=float, default=12.0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--exts", default=".mp4,.mkv,.webm,.avi")
    args = ap.parse_args()

    exts = tuple(args.exts.split(","))
    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                videos.append(os.path.relpath(os.path.join(root, fn), args.video_dir))
    print(f"found {len(videos)} videos")

    segments = []
    for rel in videos:
        dur = video_duration(os.path.join(args.video_dir, rel))
        if not dur or dur < 2.0:
            continue
        t = 0.0
        while t < dur - 1.0:
            segments.append({"video": rel, "start": round(t, 2),
                             "end": round(min(t + args.window, dur), 2), "duration": round(dur, 2)})
            t += args.stride
    random.Random(0).shuffle(segments)
    if args.max_clips:
        segments = segments[: args.max_clips]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for s in segments:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(segments)} segments -> {args.out}")
    print(f"NOTE: set train.data_root={args.video_dir} in your config.")


if __name__ == "__main__":
    main()
