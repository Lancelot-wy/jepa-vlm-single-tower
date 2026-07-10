"""Summarize V4 streaming-eval jsonls: per-arm accuracy table + paired flips vs control.

  python scripts/summarize_streaming.py <results_dir> [--control v4_ctrl_s0]

Reads every {bench}_{arm}_{mode}.jsonl in the dir (dedup by qid, keep first),
prints an accuracy table and, for each non-control arm, paired fixed/broke counts
and a two-sided sign-test p-value against the control on the common qid set.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from math import comb


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                d.setdefault(r["qid"], int(r["correct"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return d


def sign_test(a: dict, b: dict):
    common = set(a) & set(b)
    fix = sum(1 for q in common if a[q] == 1 and b[q] == 0)
    brk = sum(1 for q in common if a[q] == 0 and b[q] == 1)
    n = fix + brk
    p = min(1.0, sum(comb(n, i) for i in range(min(fix, brk) + 1)) / 2 ** n * 2) if n else 1.0
    return len(common), fix, brk, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--control", default="v4_ctrl_s0")
    args = ap.parse_args()

    runs = collections.defaultdict(dict)   # (bench, mode) -> arm -> {qid: correct}
    pat = re.compile(r"^(ovo|sb)_(.+)_(recent|prefix)\.jsonl$")
    for fn in sorted(os.listdir(args.results_dir)):
        m = pat.match(fn)
        if not m:
            continue
        bench, arm, mode = m.groups()
        runs[(bench, mode)][arm] = load(os.path.join(args.results_dir, fn))

    for (bench, mode), arms in sorted(runs.items()):
        print(f"\n=== {bench} / {mode} ===")
        print(f"{'arm':20s} {'n':>5s} {'acc':>8s}   vs {args.control}: common fixed broke p")
        ctrl = arms.get(args.control)
        for arm in sorted(arms):
            d = arms[arm]
            n = len(d)
            acc = 100 * sum(d.values()) / n if n else 0.0
            line = f"{arm:20s} {n:5d} {acc:7.2f}%"
            if ctrl is not None and arm != args.control:
                c, fx, bk, p = sign_test(d, ctrl)
                line += f"   {c:6d} {fx:5d} {bk:5d} {p:6.3f}"
            print(line)


if __name__ == "__main__":
    main()
