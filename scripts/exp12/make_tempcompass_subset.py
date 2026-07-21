#!/usr/bin/env python3
"""Create one frozen, category-stratified checkpoint-400 diagnostic subset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=120012)
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    with open(args.input) as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("任务类别") != "Tempcompass":
                continue
            row["_exp12_source_line"] = line_number
            groups[str(row.get("子类别", "unknown"))].append(row)
    rng = random.Random(args.seed)
    selected = []
    for category in sorted(groups):
        rows = list(groups[category])
        rng.shuffle(rows)
        selected.extend(rows[: args.per_category])
    selected.sort(key=lambda row: row["_exp12_source_line"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as out, open(args.ids, "w") as ids:
        for row in selected:
            line_number = row.pop("_exp12_source_line")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            ids.write(f"{line_number}\t{row.get('子类别', 'unknown')}\n")
    digest = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    with open(args.output + ".sha256", "w") as handle:
        handle.write(digest + "\n")
    print(json.dumps({"rows": len(selected), "sha256": digest,
                      "categories": {key: min(len(value), args.per_category)
                                     for key, value in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
