#!/usr/bin/env python3
"""Collect the official-budget MVBench reproduction without overstating parity."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics


PROTOCOLS = (
    "official_budget_base_full_generation",
    "official_budget_ckpt_k64_full_generation",
    "official_budget_base_cap32_generation",
    "official_budget_ckpt_k64_cap32_generation",
)
REFERENCE_PERCENT = 61.7


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def paired_test(control: dict, treatment: dict, seed: int, samples: int = 2000) -> dict:
    left = {int(row["idx"]): row for row in control["results"]}
    right = {int(row["idx"]): row for row in treatment["results"]}
    ids = sorted(left.keys() & right.keys())
    b = sum(int(left[i]["ok"] and not right[i]["ok"]) for i in ids)
    c = sum(int(not left[i]["ok"] and right[i]["ok"]) for i in ids)
    n = b + c
    if n:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
        pvalue = min(1.0, 2 * tail)
    else:
        pvalue = 1.0
    diffs = [100.0 * (int(right[i]["ok"]) - int(left[i]["ok"])) for i in ids]
    rng = random.Random(seed)
    means = []
    if diffs:
        for _ in range(samples):
            means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
        means.sort()
        ci = [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]
    else:
        ci = [0.0, 0.0]
    return {
        "paired_items": len(ids), "control_only_correct": b, "treatment_only_correct": c,
        "delta_points": statistics.mean(diffs) if diffs else 0.0,
        "paired_bootstrap_95ci": ci, "mcnemar_exact_p": pvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    documents = {
        protocol: load(os.path.join(root, f"{protocol}_mvbench.json"))
        for protocol in PROTOCOLS
    }
    rows = []
    for protocol, document in documents.items():
        if document.get("protocol") != protocol or document.get("task") != "MVBench":
            raise ValueError(f"result identity mismatch: {protocol}")
        total = int(document.get("total", 0))
        skipped = int(document.get("skipped", 0))
        if not total:
            raise ValueError(f"empty result: {protocol}")
        rows.append({
            "protocol": protocol,
            "accuracy_percent": 100.0 * float(document["acc"]),
            "correct": int(document["correct"]),
            "total": total,
            "skipped": skipped,
            "coverage_percent": 100.0 * total / max(total + skipped, 1),
            "native_video_tokens_median": document.get("metadata", {}).get(
                "native_video_tokens", {}
            ).get("median"),
        })

    raw_full = rows[0]
    delta = raw_full["accuracy_percent"] - REFERENCE_PERCENT
    abs_delta = abs(delta)
    if raw_full["coverage_percent"] < 99.5:
        parity_status = "RED_COVERAGE"
    elif abs_delta <= 2.5:
        parity_status = "GREEN_WITHIN_2_5PP"
    elif abs_delta <= 5.0:
        parity_status = "INVESTIGATE_WITHIN_5PP"
    else:
        parity_status = "RED_OUTSIDE_5PP"

    comparisons = {
        "training_effect_full": (PROTOCOLS[0], PROTOCOLS[1]),
        "frame_budget_effect_base": (PROTOCOLS[2], PROTOCOLS[0]),
        "frame_budget_effect_ckpt": (PROTOCOLS[3], PROTOCOLS[1]),
        "training_effect_cap32": (PROTOCOLS[2], PROTOCOLS[3]),
    }
    paired = {
        name: {
            "control": control,
            "treatment": treatment,
            **paired_test(documents[control], documents[treatment], 130000 + index),
        }
        for index, (name, (control, treatment)) in enumerate(comparisons.items())
    }
    output = {
        "schema_version": 1,
        "label": "official-budget reproduction; native-compatible HF runner, not private Qwen harness",
        "complete": len(rows) == 4,
        "public_reference": {"Qwen3-VL-2B-Instruct_MVBench_percent": REFERENCE_PERCENT},
        "raw_full_anchor": {
            "score_percent": raw_full["accuracy_percent"],
            "delta_from_public_points": delta,
            "coverage_percent": raw_full["coverage_percent"],
            "status": parity_status,
        },
        "rows": rows,
        "paired_tests": paired,
        "interpretation": [
            "Use raw-full to decide whether this local budget reproduction reaches the public-score band.",
            "Use full-minus-cap32 to quantify frame-budget impact under identical weights.",
            "Use checkpoint-minus-raw only within the same frame budget to estimate SFT effect.",
            "Do not rename this result an official Qwen score unless the official evaluator itself is run.",
        ],
    }
    json.dump(output, open(os.path.join(root, "official_budget_comparison.json"), "w"), indent=2)
    with open(os.path.join(root, "official_budget_comparison.md"), "w") as handle:
        handle.write("# EXP-13 official-budget MVBench anchor\n\n")
        handle.write(f"Raw/full diagnostic: **{parity_status}**; score {raw_full['accuracy_percent']:.2f}% "
                     f"vs public {REFERENCE_PERCENT:.1f}% ({delta:+.2f} pp).\n\n")
        handle.write("| protocol | accuracy | correct/total | skipped | coverage | median video tokens |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            token = row["native_video_tokens_median"]
            token_text = "—" if token is None else f"{float(token):.0f}"
            handle.write(
                f"| {row['protocol']} | {row['accuracy_percent']:.2f}% | "
                f"{row['correct']}/{row['total']} | {row['skipped']} | "
                f"{row['coverage_percent']:.2f}% | {token_text} |\n"
            )
        handle.write("\nThis is an official-budget reproduction, not the private official harness.\n")
    print(json.dumps(output["raw_full_anchor"], indent=2))


if __name__ == "__main__":
    main()
