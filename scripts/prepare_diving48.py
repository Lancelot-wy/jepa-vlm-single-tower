"""Convert Diving48 (V2 annotations) to the unified manifest.

Diving48 is a strong *temporal* probe set: 48 dive classes share near-identical
appearance and background, so probe accuracy directly reflects temporal dynamics.

Layout (http://www.svcl.ucsd.edu/projects/resound/dataset.html):
  <anno_dir>/Diving48_V2_train.json / Diving48_V2_test.json   # [{"vid_name": ..., "label": int, ...}]
  <video_dir>/<vid_name>.mp4

  python scripts/prepare_diving48.py --anno-dir /data/diving48 --video-dir /data/diving48/rgb \
      --out-dir data/diving48
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", default="data/diving48")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split, out_name in [("train", "train.jsonl"), ("test", "val.jsonl")]:
        path = os.path.join(args.anno_dir, f"Diving48_V2_{split}.json")
        with open(path) as f:
            anno = json.load(f)
        n = 0
        with open(os.path.join(args.out_dir, out_name), "w") as out:
            for a in anno:
                out.write(json.dumps({"video": f"{a['vid_name']}.mp4", "label": int(a["label"])}) + "\n")
                n += 1
        print(f"{split}: wrote {n} -> {out_name}")
    print(f"NOTE: set data_root={args.video_dir} when probing.")


if __name__ == "__main__":
    main()
