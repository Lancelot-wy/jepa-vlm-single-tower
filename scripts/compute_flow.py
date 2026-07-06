"""Annotate a manifest with a motion score for static-clip filtering (plan section 3).

Two methods:
  framediff (default) - mean |gray_t - gray_{t-1}| on 128px frames. Pure numpy, no
                        extra deps; monotone with motion; recommended on the cluster.
  farneback           - mean Farneback optical-flow magnitude (needs opencv-python).

Writes `flow` into each manifest line and prints the score distribution so you can
pick train.min_flow (suggested starting point: ~p30). Filtering itself happens at
train time via train.min_flow, so one pass serves all thresholds.

  python scripts/compute_flow.py --manifest data/llava_video/train.jsonl \
      --out data/llava_video/train_flow.jsonl --method framediff --workers 16
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# make `jepa_vlm` importable when run as `python scripts/compute_flow.py`
# (also needed inside ProcessPoolExecutor workers, which re-import this module)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gray_frames(item, data_root, size=128, num_frames=8):
    from jepa_vlm.data.video_io import decode_frames, resize_center_crop

    frames = decode_frames(
        os.path.join(data_root, item["video"]) if data_root else item["video"],
        num_frames=num_frames, sample_fps=0, sampling="uniform",
        start=item.get("start"), end=item.get("end"))
    x = resize_center_crop(frames, size).numpy()  # (T,3,S,S) in [0,1]
    return x.mean(axis=1)  # grayscale (T,S,S)


def score_framediff(task):
    item, data_root = task
    try:
        g = _gray_frames(item, data_root)
        # mean absolute frame difference, scaled x100 for readable magnitudes
        item["flow"] = round(float(np.abs(np.diff(g, axis=0)).mean() * 100), 4)
    except Exception as e:
        print(f"framediff failed for {item['video']}: {e}")
        item["flow"] = None
    return item


def score_farneback(task):
    item, data_root = task
    try:
        import cv2

        g = (_gray_frames(item, data_root) * 255).astype(np.uint8)
        mags = []
        for a, b in zip(g[:-1], g[1:]):
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.linalg.norm(flow, axis=-1).mean()))
        item["flow"] = round(float(np.mean(mags)), 4)
    except Exception as e:
        print(f"farneback failed for {item['video']}: {e}")
        item["flow"] = None
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="framediff", choices=["framediff", "farneback"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="skip videos already present in --out and append")
    args = ap.parse_args()

    if args.method == "farneback":
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("opencv not installed -> falling back to framediff")
            args.method = "framediff"
    score = score_framediff if args.method == "framediff" else score_farneback

    with open(args.manifest) as f:
        items = [json.loads(l) for l in f if l.strip()]

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    done.add(json.loads(l)["video"])
                except Exception:
                    continue
        print(f"resume: {len(done)} already scored, skipping them")
    pending = [it for it in items if it["video"] not in done]
    tasks = [(it, args.data_root) for it in pending]
    open_mode = "a" if (args.resume and done) else "w"
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, open_mode) as out:
        for i, item in enumerate(ex.map(score, tasks, chunksize=16)):
            out.write(json.dumps(item) + "\n")
            out.flush()
            if i % 1000 == 0:
                print(f"{i}/{len(tasks)} (pending)")

    with open(args.out) as f:
        flows = np.array([json.loads(l)["flow"] for l in f
                          if l.strip() and json.loads(l).get("flow") is not None])
    print(f"\ndone: {len(flows)}/{len(items)} scored (method={args.method})")
    if len(flows):
        qs = [5, 10, 20, 30, 50, 70, 90]
        print("flow percentiles:")
        for q, v in zip(qs, np.percentile(flows, qs)):
            print(f"  p{q:<3d} {v:.3f}")
        p30 = float(np.percentile(flows, 30))
        print(f"suggested starting threshold: train.min_flow={p30:.2f} (~p30, "
              f"drops the most static ~30%); inspect a few clips around it before committing.")


if __name__ == "__main__":
    main()
