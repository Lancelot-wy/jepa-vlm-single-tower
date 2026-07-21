#!/usr/bin/env python3
"""Materialize disabled B0-B5 templates after a best K has been selected."""

from __future__ import annotations

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-k", type=int, choices=(4, 16, 64), required=True)
    parser.add_argument("--source", default="configs/orca_event")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output or f"configs/orca_event_k{args.best_k}")
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob("b*.yaml")):
        text = path.read_text()
        if "PLACEHOLDER_K" not in text:
            raise SystemExit(f"template lacks PLACEHOLDER_K: {path}")
        (output / path.name).write_text(text.replace("PLACEHOLDER_K", str(args.best_k)))
        count += 1
    if count != 6:
        raise SystemExit(f"expected 6 templates, found {count}")
    print(f"materialized {count} configs in {output}; no jobs were submitted")


if __name__ == "__main__":
    main()
