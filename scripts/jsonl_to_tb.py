#!/usr/bin/env python3
"""Convert jepa-vlm log.jsonl files into TensorBoard event files.

Non-destructive: reads each run's <output_dir>/log.jsonl and (re)writes
<output_dir>/tb/*.  Safe to re-run while a job is still training to refresh
the curves.  Native tb logging is also emitted by train.py for new runs; this
script exists so already-running / completed runs get a unified view too.

Usage:
  python scripts/jsonl_to_tb.py OUTPUTS_ROOT [run_dir ...]
  # default: scan every immediate subdir of OUTPUTS_ROOT that has a log.jsonl
"""
from __future__ import annotations

import json
import os
import sys

from torch.utils.tensorboard import SummaryWriter


def convert(run_dir: str) -> int:
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.isfile(log_path):
        return 0
    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    n = 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            # val records only carry eval keys; train records carry lr/sec_per_step.
            tag = "train" if ("lr" in rec or "sec_per_step" in rec) else "val"
            for k, v in rec.items():
                if k == "step" or not isinstance(v, (int, float)):
                    continue
                writer.add_scalar(f"{tag}/{k}", v, step)
            n += 1
    writer.close()
    return n


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    root = argv[1]
    runs = argv[2:] or [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, d, "log.jsonl"))
    ]
    for r in runs:
        n = convert(r)
        print(f"{r}: {n} records -> {os.path.join(r, 'tb')}")


if __name__ == "__main__":
    main(sys.argv)
