#!/usr/bin/env python3
"""Collect the EXP-12 native-Qwen anchor matrix and paired protocol deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics


PROTOCOLS = (
    "custom_base_k4_full_option",
    "custom_base_k16_full_option",
    "custom_base_k64_full_option",
    "custom_ckpt_k64_full_option",
    "custom_base_k64_letter",
    "custom_ckpt_k64_letter",
    "native_base_matched32_generation",
    "native_ckpt_k64_matched32_generation",
)
TASKS = ("MVBench", "Tempcompass")


def task_slug(task: str) -> str:
    return task.lower()


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def paired_test(control: dict, treatment: dict, seed: int, samples: int = 2000) -> dict:
    left = {int(row["idx"]): row for row in control["results"]}
    right = {int(row["idx"]): row for row in treatment["results"]}
    indices = sorted(left.keys() & right.keys())
    b = sum(int(left[index]["ok"] and not right[index]["ok"]) for index in indices)
    c = sum(int(not left[index]["ok"] and right[index]["ok"]) for index in indices)
    discordant = b + c
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2 ** discordant)
        pvalue = min(1.0, 2 * tail)
    else:
        pvalue = 1.0
    differences = [
        100.0 * (int(right[index]["ok"]) - int(left[index]["ok"]))
        for index in indices
    ]
    rng = random.Random(seed)
    bootstraps = []
    if differences:
        for _ in range(samples):
            bootstraps.append(
                sum(differences[rng.randrange(len(differences))] for _ in differences)
                / len(differences)
            )
        bootstraps.sort()
        confidence_interval = [
            bootstraps[int(0.025 * samples)],
            bootstraps[int(0.975 * samples) - 1],
        ]
    else:
        confidence_interval = [0.0, 0.0]
    return {
        "paired_items": len(indices),
        "control_only_correct": b,
        "treatment_only_correct": c,
        "delta_points": statistics.mean(differences) if differences else 0.0,
        "mcnemar_exact_p": pvalue,
        "paired_bootstrap_95ci": confidence_interval,
    }


def native_token_median(document: dict) -> float | None:
    value = document.get("metadata", {}).get("native_video_tokens", {}).get("median")
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    documents: dict[tuple[str, str], dict] = {}
    missing = []
    for protocol in PROTOCOLS:
        for task in TASKS:
            path = os.path.join(root, f"{protocol}_{task_slug(task)}.json")
            if not os.path.isfile(path):
                missing.append(path)
                continue
            document = load_json(path)
            if document.get("protocol") != protocol or document.get("task") != task:
                raise ValueError(f"protocol/task mismatch in {path}")
            if not document.get("total"):
                raise ValueError(f"empty evaluation: {path}")
            documents[(protocol, task)] = document
    if missing and not args.allow_incomplete:
        raise SystemExit("missing required results:\n" + "\n".join(missing))

    rows = []
    for (protocol, task), document in documents.items():
        rows.append({
            "protocol": protocol,
            "task": task,
            "accuracy_percent": 100.0 * float(document["acc"]),
            "correct": int(document["correct"]),
            "total": int(document["total"]),
            "skipped": int(document.get("skipped", 0)),
            "scoring": document.get("scoring"),
            "native_video_tokens_median": native_token_median(document),
        })
    rows.sort(key=lambda row: (PROTOCOLS.index(row["protocol"]), TASKS.index(row["task"])))

    comparisons = {
        "custom_base_k64_minus_k4": (
            "custom_base_k4_full_option", "custom_base_k64_full_option"
        ),
        "custom_training_effect_k64_full_option": (
            "custom_base_k64_full_option", "custom_ckpt_k64_full_option"
        ),
        "custom_training_effect_k64_letter": (
            "custom_base_k64_letter", "custom_ckpt_k64_letter"
        ),
        "custom_scorer_sensitivity_base_k64": (
            "custom_base_k64_full_option", "custom_base_k64_letter"
        ),
        "native_training_effect_k64": (
            "native_base_matched32_generation", "native_ckpt_k64_matched32_generation"
        ),
        "native_minus_custom_base_k64": (
            "custom_base_k64_letter", "native_base_matched32_generation"
        ),
        "native_minus_custom_ckpt_k64": (
            "custom_ckpt_k64_letter", "native_ckpt_k64_matched32_generation"
        ),
    }
    paired = {}
    for comparison_index, (name, (control, treatment)) in enumerate(comparisons.items()):
        for task_index, task in enumerate(TASKS):
            if (control, task) not in documents or (treatment, task) not in documents:
                continue
            paired[f"{name}:{task}"] = {
                "control": control,
                "treatment": treatment,
                **paired_test(
                    documents[(control, task)],
                    documents[(treatment, task)],
                    seed=131300 + 100 * comparison_index + task_index,
                ),
            }

    output = {
        "schema_version": 1,
        "complete": not missing,
        "missing": missing,
        "rows": rows,
        "paired_tests": paired,
        "interpretation_order": [
            "First establish the raw Qwen base score under every protocol.",
            "Use trained-minus-base within one protocol as the SFT effect.",
            "Use K64-minus-K4 only inside the historical custom protocol as the K effect.",
            "Do not compare these absolute scores directly with Qwen's published 61.7.",
        ],
    }
    json_path = os.path.join(root, "native_anchor_comparison.json")
    csv_path = os.path.join(root, "native_anchor_comparison.csv")
    markdown_path = os.path.join(root, "native_anchor_comparison.md")
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    fields = list(rows[0]) if rows else []
    if fields:
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with open(markdown_path, "w") as handle:
        handle.write("# EXP-13 / EXP-12 native-Qwen evaluation anchor\n\n")
        handle.write(f"Complete: **{not missing}**\n\n")
        handle.write("| protocol | task | accuracy | correct/total | skipped | native tokens (median) |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            token_value = row["native_video_tokens_median"]
            token_text = "—" if token_value is None else f"{token_value:.0f}"
            handle.write(
                f"| {row['protocol']} | {row['task']} | {row['accuracy_percent']:.2f}% | "
                f"{row['correct']}/{row['total']} | {row['skipped']} | {token_text} |\n"
            )
        handle.write("\n## Paired deltas\n\n")
        handle.write("| comparison | delta (pp) | 95% CI | McNemar p | paired N |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for name, result in paired.items():
            low, high = result["paired_bootstrap_95ci"]
            handle.write(
                f"| {name} | {result['delta_points']:.3f} | "
                f"[{low:.3f}, {high:.3f}] | {result['mcnemar_exact_p']:.4g} | "
                f"{result['paired_items']} |\n"
            )
        handle.write("\nAbsolute scores from different protocols are diagnostic, not interchangeable.\n")
    print(json.dumps({
        "complete": not missing,
        "rows": len(rows),
        "paired_tests": len(paired),
        "json": json_path,
        "markdown": markdown_path,
    }, indent=2))


if __name__ == "__main__":
    main()
