#!/usr/bin/env python3
"""Build a cross-run comparison table for jepa-vlm experiments.

Scans each run's <output_dir>/log.jsonl, takes the latest train record (and the
latest val record if present), and emits a unified comparison table as both CSV
and Markdown under <outputs_root>/summary/.  Re-run anytime to refresh (safe
while jobs are still training).

Usage:
  python scripts/summarize_runs.py OUTPUTS_ROOT
"""
from __future__ import annotations

import csv
import json
import os
import sys

# columns shown in the table (train-side metrics); order matters.
TRAIN_COLS = ["step", "loss", "reg_loss", "mtp_loss", "target_std", "adj_cos",
              "copy_mse", "sec_per_step"]
# val records store keys with a "val/" prefix (see train.py quick_eval).
# nontrivial_ratio is the plan's go/no-go gate - keep it in the table.
VAL_COLS = ["reg_loss", "copy_mse", "nontrivial_ratio", "target_std", "adj_cos"]


def last_records(log_path: str):
    last_train, last_val = None, None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "lr" in rec or "sec_per_step" in rec:
                last_train = rec
            else:
                last_val = rec
    return last_train, last_val


def fmt(v):
    if isinstance(v, float):
        return round(v, 5)
    return v


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    root = argv[1]
    runs = sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "log.jsonl")))
    rows = []
    for d in runs:
        name = d.replace("jepa_llava_video_", "")
        tr, va = last_records(os.path.join(root, d, "log.jsonl"))
        row = {"run": name}
        for c in TRAIN_COLS:
            row[c] = fmt(tr.get(c)) if tr else None
        for c in VAL_COLS:
            row[f"val_{c}"] = fmt(va.get(f"val/{c}")) if va else None
        rows.append(row)

    out_dir = os.path.join(root, "summary")
    os.makedirs(out_dir, exist_ok=True)
    header = ["run"] + TRAIN_COLS + [f"val_{c}" for c in VAL_COLS]

    csv_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, "comparison.md")
    with open(md_path, "w") as f:
        f.write("# jepa-vlm run comparison (latest logged step)\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in header) + " |\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}\n")
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main(sys.argv)
