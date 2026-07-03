"""Annotate a manifest with mean optical-flow magnitude (static-clip filtering, plan section 3).

Farneback flow on 8 fps, 128px grayscale frames; writes `flow` into each manifest line.
Filtering happens at train time via train.min_flow (so one pass serves all thresholds).

  python scripts/compute_flow.py --manifest data/ssv2/train.jsonl --data-root /data/ssv2/videos \
      --out data/ssv2/train_flow.jsonl --workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def mean_flow(task):
    item, data_root = task
    try:
        import cv2
        from jepa_vlm.data.video_io import decode_frames

        frames = decode_frames(
            os.path.join(data_root, item["video"]), num_frames=8, sample_fps=0,
            sampling="uniform", start=item.get("start"), end=item.get("end"))
        gray = [cv2.cvtColor(cv2.resize(f, (128, 128)), cv2.COLOR_RGB2GRAY) for f in frames]
        mags = []
        for a, b in zip(gray[:-1], gray[1:]):
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.linalg.norm(flow, axis=-1).mean()))
        item["flow"] = round(float(np.mean(mags)), 4)
    except Exception as e:
        print(f"flow failed for {item['video']}: {e}")
        item["flow"] = None
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.manifest) as f:
        items = [json.loads(l) for l in f if l.strip()]
    tasks = [(it, args.data_root) for it in items]
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, "w") as out:
        for i, item in enumerate(ex.map(mean_flow, tasks, chunksize=16)):
            out.write(json.dumps(item) + "\n")
            if i % 1000 == 0:
                print(f"{i}/{len(items)}")
    flows = [it["flow"] for it in items if it.get("flow")]
    print(f"done. Suggested min_flow ~ p20 of distribution once computed; "
          f"inspect percentiles with numpy on the 'flow' field.")


if __name__ == "__main__":
    main()
