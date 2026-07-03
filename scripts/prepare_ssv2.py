"""Convert Something-Something-V2 official annotations to the unified manifest.

Expected layout (official 20BN/Qualcomm release):
  <anno_dir>/labels.json                            # {"template with [x]": "id", ...} 174 classes
  <anno_dir>/train.json / validation.json           # [{"id": "42", "template": "...", ...}]
  <video_dir>/<id>.webm

  python scripts/prepare_ssv2.py --anno-dir /data/ssv2/anno --video-dir /data/ssv2/videos \
      --out-dir data/ssv2
"""

import argparse
import json
import os


def norm_template(t: str) -> str:
    return t.replace("[something]", "something").replace("[", "").replace("]", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", required=True)
    ap.add_argument("--video-dir", required=True, help="dir with <id>.webm; stored relative to --data-root at train time")
    ap.add_argument("--out-dir", default="data/ssv2")
    ap.add_argument("--check-files", action="store_true", help="drop entries whose video file is missing")
    args = ap.parse_args()

    with open(os.path.join(args.anno_dir, "labels.json")) as f:
        label2id = {norm_template(k): int(v) for k, v in json.load(f).items()}

    os.makedirs(args.out_dir, exist_ok=True)
    for split, out_name in [("train", "train.jsonl"), ("validation", "val.jsonl")]:
        with open(os.path.join(args.anno_dir, f"{split}.json")) as f:
            anno = json.load(f)
        n_written = n_missing = 0
        with open(os.path.join(args.out_dir, out_name), "w") as out:
            for a in anno:
                rel = f"{a['id']}.webm"
                if args.check_files and not os.path.exists(os.path.join(args.video_dir, rel)):
                    n_missing += 1
                    continue
                tpl = norm_template(a["template"])
                out.write(json.dumps({
                    "video": rel,
                    "label": label2id[tpl],
                    "label_name": tpl,
                }) + "\n")
                n_written += 1
        print(f"{split}: wrote {n_written} (missing {n_missing}) -> {out_name}")
    print(f"classes: {len(label2id)}")
    print(f"NOTE: set train.data_root={args.video_dir} in your config.")


if __name__ == "__main__":
    main()
