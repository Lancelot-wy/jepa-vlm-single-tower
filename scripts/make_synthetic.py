"""Generate a tiny synthetic video dataset (moving square; label = motion direction).

Used by the smoke test and as a mechanism sanity-check: direction classes need motion,
shuffle/reverse probes need order. Not a substitute for SSv2.

  python scripts/make_synthetic.py --out data/synthetic --num 32
"""

import argparse
import json
import os

import numpy as np

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up


def render(path: str, direction: int, size=128, frames=24, fps=8, seed=0):
    import av

    rng = np.random.default_rng(seed)
    dx, dy = DIRS[direction]
    sq = 24
    speed = 3.5
    x = rng.integers(sq, size - 2 * sq) if dx <= 0 else sq
    y = rng.integers(sq, size - 2 * sq) if dy <= 0 else sq
    if dx < 0:
        x = size - 2 * sq
    if dy < 0:
        y = size - 2 * sq
    color = rng.integers(80, 255, 3)
    bg = rng.integers(0, 60, 3)

    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for t in range(frames):
            img = np.ones((size, size, 3), np.uint8) * bg.astype(np.uint8)
            xi, yi = int(x + dx * speed * t), int(y + dy * speed * t)
            xi, yi = np.clip(xi, 0, size - sq), np.clip(yi, 0, size - sq)
            img[yi : yi + sq, xi : xi + sq] = color
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.5)
    args = ap.parse_args()

    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    items = []
    for i in range(args.num):
        d = i % 4
        rel = f"videos/clip_{i:04d}.mp4"
        render(os.path.join(args.out, rel), d, seed=i)
        items.append({"video": rel, "label": d, "label_name": ["right", "left", "down", "up"][d]})
    n_val = int(len(items) * args.val_frac)
    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        for it in items[n_val:]:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(args.out, "val.jsonl"), "w") as f:
        for it in items[:n_val]:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items) - n_val} train / {n_val} val clips under {args.out}")


if __name__ == "__main__":
    main()
