#!/usr/bin/env python3
"""Collect EXP-14 and join it to the frozen EXP-12 K64 controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics


NEW_ARMS = {
    "b0_ce_seed1": (1, "ce"),
    "b1_query_seed1": (1, "query"),
    "b2_noquery_seed0": (0, "no_query"),
    "b3_noquery_seed1": (1, "no_query"),
    "b4_query_beatcopy_seed0": (0, "query_beatcopy"),
    "b5_query_beatcopy_seed1": (1, "query_beatcopy"),
}
SOURCE_ARMS = {
    "a4_ce_k64": (0, "ce"),
    "a5_query_k64": (0, "query"),
}
TASK_FILES = {
    "MVBench": "checkpoint-800_mvbench.json",
    "TempCompass": "checkpoint-800_tempcompass.json",
}


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def checkpoint_complete(root: str, step: int = 800) -> bool:
    checkpoint = os.path.join(root, f"checkpoint-{step}")
    state = os.path.join(checkpoint, "state.pt")
    meta = os.path.join(checkpoint, "checkpoint_meta.json")
    if not os.path.isfile(state) or not os.path.isfile(meta):
        return False
    value = load_json(meta)
    return bool(
        value.get("step") == step
        and value.get("step_unit") == "optimizer_update"
        and value.get("state_bytes") == os.path.getsize(state)
    )


def training_rows(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [row for row in rows if "loss" in row]
    if not rows:
        raise ValueError(f"no training rows: {path}")
    return rows


def paired_test(control: dict, treatment: dict, seed: int, samples: int = 2000) -> dict:
    left = {int(row["idx"]): row for row in control["results"]}
    right = {int(row["idx"]): row for row in treatment["results"]}
    indices = sorted(left.keys() & right.keys())
    b = sum(int(left[i]["ok"] and not right[i]["ok"]) for i in indices)
    c = sum(int(not left[i]["ok"] and right[i]["ok"]) for i in indices)
    discordant = b + c
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / 2**discordant
        pvalue = min(1.0, 2 * tail)
    else:
        pvalue = 1.0
    differences = [100.0 * (int(right[i]["ok"]) - int(left[i]["ok"])) for i in indices]
    rng = random.Random(seed)
    bootstraps = []
    if differences:
        for _ in range(samples):
            bootstraps.append(
                sum(differences[rng.randrange(len(differences))] for _ in differences)
                / len(differences)
            )
        bootstraps.sort()
        ci = [bootstraps[int(0.025 * samples)], bootstraps[int(0.975 * samples) - 1]]
    else:
        ci = [0.0, 0.0]
    return {
        "paired_items": len(indices),
        "control_only_correct": b,
        "treatment_only_correct": c,
        "delta_points": statistics.mean(differences) if differences else 0.0,
        "mcnemar_exact_p": pvalue,
        "paired_bootstrap_95ci": ci,
    }


def read_text(path: str) -> str:
    return open(path).read().strip() if os.path.isfile(path) else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="EXP-14 exp14_state_diagnostics root")
    parser.add_argument("--source", required=True, help="frozen EXP-12 token-sweep root")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    source = os.path.abspath(args.source)
    scorecard_root = os.path.join(root, "joined_scorecards")
    os.makedirs(scorecard_root, exist_ok=True)

    rows = []
    documents: dict[tuple[int, str, str], dict] = {}
    arm_roots = {
        **{arm: (os.path.join(source, arm), seed, mode, "EXP-12")
           for arm, (seed, mode) in SOURCE_ARMS.items()},
        **{arm: (os.path.join(root, arm), seed, mode, "EXP-14")
           for arm, (seed, mode) in NEW_ARMS.items()},
    }
    for arm, (arm_root, seed, mode, origin) in arm_roots.items():
        evaluations = {
            task: load_json(os.path.join(arm_root, filename))
            for task, filename in TASK_FILES.items()
        }
        logs = training_rows(os.path.join(arm_root, "trainer_log.jsonl"))
        last = logs[-1]
        numeric = [value for row in logs for value in row.values() if isinstance(value, (int, float))]
        row = {
            "arm": arm,
            "origin": origin,
            "seed": seed,
            "mode": mode,
            "K": 64,
            "MVBench": 100.0 * float(evaluations["MVBench"]["acc"]),
            "TempCompass": 100.0 * float(evaluations["TempCompass"]["acc"]),
            "ce_loss": last.get("ce_loss"),
            "centered_margin": last.get("state/centered_margin"),
            "persistence_ratio": last.get("state/persistence_ratio"),
            "beat_copy_loss": last.get("state/beat_copy_loss"),
            "retrieval_top1": last.get("state/retrieval_top1"),
            "retrieval_top5": last.get("state/retrieval_top5"),
            "samples_per_sec": last.get("samples_per_sec"),
            "max_memory_gb": max(float(item.get("max_memory_gb", 0)) for item in logs),
            "checkpoint_complete": checkpoint_complete(arm_root),
            "evaluator_complete": all(document.get("total", 0) > 0 for document in evaluations.values()),
            "finite_training": all(math.isfinite(float(value)) for value in numeric),
            "manifest_sha256": read_text(os.path.join(arm_root, "manifest.sha256")),
            "training_commit": read_text(os.path.join(arm_root, "git_commit.txt")),
        }
        predictive = mode != "ce"
        row["mechanism_gate"] = bool(
            not predictive
            or (
                row["centered_margin"] is not None
                and float(row["centered_margin"]) > 0.10
                and row["persistence_ratio"] is not None
                and float(row["persistence_ratio"]) < 0.90
            )
        )
        rows.append(row)
        for task, document in evaluations.items():
            documents[(seed, mode, task)] = document

    manifests = {row["manifest_sha256"] for row in rows}
    new_commits = {row["training_commit"] for row in rows if row["origin"] == "EXP-14"}
    data_consistent = len(manifests) == 1 and "" not in manifests
    new_code_consistent = len(new_commits) == 1 and "" not in new_commits
    row_by_key = {(row["seed"], row["mode"]): row for row in rows}
    for row in rows:
        control = row_by_key[(row["seed"], "ce")]
        row["MVBench_vs_same_seed_ce"] = row["MVBench"] - control["MVBench"]
        row["TempCompass_vs_same_seed_ce"] = row["TempCompass"] - control["TempCompass"]
        row["benchmark_protection_gate"] = bool(
            row["MVBench_vs_same_seed_ce"] >= -1.0
            and row["TempCompass_vs_same_seed_ce"] >= -1.0
        )
        row["candidate_gate"] = bool(
            row["mode"] != "ce"
            and row["mechanism_gate"]
            and row["benchmark_protection_gate"]
        )
        row["data_consistent"] = data_consistent
        row["new_code_consistent"] = new_code_consistent
        with open(os.path.join(scorecard_root, row["arm"] + ".json"), "w") as handle:
            json.dump(row, handle, indent=2)

    comparisons = []
    for seed in (0, 1):
        comparisons.extend([
            (f"seed{seed}_query_minus_ce", seed, "ce", "query"),
            (f"seed{seed}_noquery_minus_ce", seed, "ce", "no_query"),
            (f"seed{seed}_noquery_minus_query", seed, "query", "no_query"),
            (f"seed{seed}_beatcopy_minus_query", seed, "query", "query_beatcopy"),
        ])
    for mode in ("ce", "query", "no_query", "query_beatcopy"):
        comparisons.append((f"{mode}_seed1_minus_seed0", None, mode, mode))

    paired = {}
    for comparison_index, (name, seed, control_mode, treatment_mode) in enumerate(comparisons):
        for task_index, task in enumerate(TASK_FILES):
            if seed is None:
                control_key = (0, control_mode, task)
                treatment_key = (1, treatment_mode, task)
            else:
                control_key = (seed, control_mode, task)
                treatment_key = (seed, treatment_mode, task)
            paired[f"{name}:{task}"] = {
                "control": list(control_key[:2]),
                "treatment": list(treatment_key[:2]),
                **paired_test(
                    documents[control_key], documents[treatment_key],
                    seed=140000 + 100 * comparison_index + task_index,
                ),
            }

    aggregate = {}
    for mode in ("query", "no_query", "query_beatcopy"):
        aggregate[mode] = {
            task: statistics.mean(
                row_by_key[(seed, mode)][task] - row_by_key[(seed, "ce")][task]
                for seed in (0, 1)
            )
            for task in TASK_FILES
        }
    output = {
        "schema_version": 1,
        "complete": data_consistent and new_code_consistent and all(
            row["checkpoint_complete"] and row["evaluator_complete"] and row["finite_training"]
            for row in rows
        ),
        "data_consistent": data_consistent,
        "new_code_consistent": new_code_consistent,
        "source": source,
        "rows": rows,
        "mean_delta_vs_same_seed_ce": aggregate,
        "paired_tests": paired,
        "decision": {
            "auto_event_submission": False,
            "reason": "EXP-14 diagnoses query/no-query and anti-copy behavior; Event remains a manual decision.",
            "mechanism_thresholds": {"centered_margin_gt": 0.10, "persistence_ratio_lt": 0.90},
            "benchmark_protection_tolerance_points": -1.0,
        },
    }
    with open(os.path.join(root, "comparison.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    fields = list(rows[0])
    with open(os.path.join(root, "comparison.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(root, "comparison.md"), "w") as handle:
        handle.write("# EXP-14 K64 state-mechanism diagnostics\n\n")
        handle.write("| arm | seed | mode | MVBench | TempCompass | margin | persistence | mechanism | protection | candidate |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['arm']} | {row['seed']} | {row['mode']} | {row['MVBench']:.2f} | "
                f"{row['TempCompass']:.2f} | {row['centered_margin'] or 0:.4f} | "
                f"{row['persistence_ratio'] or 0:.4f} | {row['mechanism_gate']} | "
                f"{row['benchmark_protection_gate']} | {row['candidate_gate']} |\n"
            )
        handle.write("\nEvent-conditioned training is not auto-submitted by this collector.\n")
    print(json.dumps({"complete": output["complete"], "root": root, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
