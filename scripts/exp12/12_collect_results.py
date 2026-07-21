#!/usr/bin/env python3
"""Collect EXP-12 artifacts, paired statistics, and K/mode deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics


ARMS = {
    "a0_ce_k4": (4, "none"), "a1_query_k4": (4, "query"),
    "a2_ce_k16": (16, "none"), "a3_query_k16": (16, "query"),
    "a4_ce_k64": (64, "none"), "a5_query_k64": (64, "query"),
}


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def checkpoint_complete(root: str, step: int) -> bool:
    checkpoint = os.path.join(root, f"checkpoint-{step}")
    state_path = os.path.join(checkpoint, "state.pt")
    meta_path = os.path.join(checkpoint, "checkpoint_meta.json")
    if not os.path.isfile(state_path) or not os.path.isfile(meta_path):
        return False
    meta = load_json(meta_path)
    return bool(
        meta.get("step") == step
        and meta.get("step_unit") == "optimizer_update"
        and meta.get("state_bytes") == os.path.getsize(state_path)
    )


def categories(document):
    grouped = {}
    for row in document["results"]:
        key = str(row.get("sub_type", "unknown")).strip().lower().replace("_", " ")
        grouped.setdefault(key, [0, 0])
        grouped[key][0] += int(row["ok"])
        grouped[key][1] += 1
    return {key: 100 * good / max(total, 1) for key, (good, total) in grouped.items()}


def last_training_metrics(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    train = [row for row in rows if "loss" in row]
    if not train:
        raise ValueError(f"no training metrics in {path}")
    return train, train[-1]


def rank_slope(rows):
    values = [row.get("state/target_effective_rank") for row in rows]
    values = [float(value) for value in values if value is not None]
    if len(values) < 3:
        return 0.0
    values = values[len(values) // 2:]
    xbar = (len(values) - 1) / 2
    ybar = statistics.mean(values)
    denom = sum((index - xbar) ** 2 for index in range(len(values)))
    return sum((index - xbar) * (value - ybar) for index, value in enumerate(values)) / max(denom, 1)


def paired_rows(left, right):
    left_by_id = {row["idx"]: row for row in left["results"]}
    right_by_id = {row["idx"]: row for row in right["results"]}
    ids = sorted(left_by_id.keys() & right_by_id.keys())
    return [(left_by_id[index], right_by_id[index]) for index in ids]


def mcnemar_and_bootstrap(control, treatment, seed=120012, samples=2000):
    pairs = paired_rows(control, treatment)
    b = sum(1 for left, right in pairs if left["ok"] and not right["ok"])
    c = sum(1 for left, right in pairs if not left["ok"] and right["ok"])
    discordant = b + c
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2 ** discordant)
        pvalue = min(1.0, 2 * tail)
    else:
        pvalue = 1.0
    diffs = [100.0 * (right["ok"] - left["ok"]) for left, right in pairs]
    rng = random.Random(seed)
    means = []
    if diffs:
        for _ in range(samples):
            means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
        means.sort()
        ci = [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]
    else:
        ci = [0.0, 0.0]
    return {"paired_items": len(pairs), "control_only_correct": b,
            "treatment_only_correct": c, "mcnemar_exact_p": pvalue,
            "delta_points": statistics.mean(diffs) if diffs else 0.0,
            "paired_bootstrap_95ci": ci}


def write_jsonl(document, path):
    with open(path, "w") as handle:
        for row in document["results"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    rows = []
    raw = {}
    for arm, (k, mode) in ARMS.items():
        arm_root = os.path.join(root, arm)
        mv = load_json(os.path.join(arm_root, "checkpoint-800_mvbench.json"))
        tc = load_json(os.path.join(arm_root, "checkpoint-800_tempcompass.json"))
        train_rows, last = last_training_metrics(os.path.join(arm_root, "trainer_log.jsonl"))
        cat = categories(tc)
        mechanism = {
            key: value for key, value in last.items()
            if key.startswith("state/") or key.startswith("model/") or key.startswith("video/")
        }
        mechanism["state/target_effective_rank_slope"] = rank_slope(train_rows)
        with open(os.path.join(arm_root, "mechanism_metrics.json"), "w") as handle:
            json.dump(mechanism, handle, indent=2)
        write_jsonl(mv, os.path.join(arm_root, "mvbench_predictions.jsonl"))
        write_jsonl(tc, os.path.join(arm_root, "tempcompass_predictions.jsonl"))
        with open(os.path.join(arm_root, "mvbench_categories.json"), "w") as handle:
            json.dump(categories(mv), handle, indent=2)
        with open(os.path.join(arm_root, "tempcompass_categories.json"), "w") as handle:
            json.dump(cat, handle, indent=2)
        wall = sum(
            float(item.get("sec_per_step", 0)) * int(item.get("interval_steps", 10))
            for item in train_rows
        )
        numeric_values = [
            value for item in train_rows for value in item.values()
            if isinstance(value, (int, float))
        ]
        manifest_path = os.path.join(arm_root, "manifest.sha256")
        manifest_sha = ""
        if os.path.isfile(manifest_path):
            with open(manifest_path) as handle:
                manifest_sha = handle.read().strip()
        commit_path = os.path.join(arm_root, "git_commit.txt")
        training_commit = ""
        if os.path.isfile(commit_path):
            with open(commit_path) as handle:
                training_commit = handle.read().strip()
        row = {
            "arm": arm, "K": k, "mode": mode,
            "MVBench": 100 * mv["acc"], "TempCompass": 100 * tc["acc"],
            "action": cat.get("action"), "direction": cat.get("direction"),
            "order": cat.get("order"), "speed": cat.get("speed"),
            "attribute_change": cat.get("attribute change", cat.get("attribute_change")),
            "ce_loss": last.get("ce_loss"),
            "centered_margin": last.get("state/centered_margin"),
            "persistence_ratio": last.get("state/persistence_ratio"),
            "dynamic_fraction": last.get("state/dynamic_sample_fraction"),
            "target_effective_rank": last.get("state/target_effective_rank"),
            "target_effective_rank_slope": mechanism["state/target_effective_rank_slope"],
            "retrieval_top1": last.get("state/retrieval_top1"),
            "retrieval_top5": last.get("state/retrieval_top5"),
            "samples_per_sec": last.get("samples_per_sec"),
            "max_memory_gb": max(float(item.get("max_memory_gb", 0)) for item in train_rows),
            "wall_time_sec": wall,
            "checkpoint_complete": checkpoint_complete(arm_root, 800),
            "evaluator_complete": bool(mv.get("total", 0) and tc.get("total", 0)),
            "finite_training": all(math.isfinite(float(value)) for value in numeric_values),
            "manifest_sha256": manifest_sha,
            "training_commit": training_commit,
        }
        with open(os.path.join(arm_root, "scorecard.json"), "w") as handle:
            json.dump(row, handle, indent=2)
        rows.append(row)
        raw[arm] = {"mv": mv, "tc": tc}

    manifests = {row["manifest_sha256"] for row in rows}
    commits = {row["training_commit"] for row in rows}
    data_consistent = len(manifests) == 1 and "" not in manifests
    code_consistent = len(commits) == 1 and "" not in commits
    for row in rows:
        row["data_consistent"] = data_consistent
        row["code_consistent"] = code_consistent
        with open(os.path.join(root, row["arm"], "scorecard.json"), "w") as handle:
            json.dump(row, handle, indent=2)
    by_k = {k: {row["mode"]: row for row in rows if row["K"] == k} for k in (4, 16, 64)}
    deltas = {}
    paired = {}
    for k in (4, 16, 64):
        control, query = by_k[k]["none"], by_k[k]["query"]
        deltas[f"query_minus_ce_k{k}"] = {
            metric: (query[metric] - control[metric])
            for metric in ("MVBench", "TempCompass", "order", "direction", "speed")
            if query.get(metric) is not None and control.get(metric) is not None
        }
        paired[f"k{k}_MVBench"] = mcnemar_and_bootstrap(
            raw[control["arm"]]["mv"], raw[query["arm"]]["mv"], seed=120012 + k
        )
        paired[f"k{k}_TempCompass"] = mcnemar_and_bootstrap(
            raw[control["arm"]]["tc"], raw[query["arm"]]["tc"], seed=130012 + k
        )
    deltas["ce_k16_minus_k4"] = {
        metric: by_k[16]["none"][metric] - by_k[4]["none"][metric]
        for metric in ("MVBench", "TempCompass")
    }
    deltas["ce_k64_minus_k16"] = {
        metric: by_k[64]["none"][metric] - by_k[16]["none"][metric]
        for metric in ("MVBench", "TempCompass")
    }
    document = {"rows": rows, "deltas": deltas, "paired_tests": paired}
    with open(os.path.join(root, "comparison.json"), "w") as handle:
        json.dump(document, handle, indent=2)
    fields = list(rows[0])
    with open(os.path.join(root, "comparison.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(root, "comparison.md"), "w") as handle:
        handle.write("# EXP-12 Orca visual-token sweep\n\n")
        cols = ("arm", "K", "mode", "MVBench", "TempCompass", "order", "direction", "speed",
                "ce_loss", "centered_margin", "persistence_ratio", "dynamic_fraction",
                "retrieval_top1", "retrieval_top5", "samples_per_sec", "max_memory_gb",
                "wall_time_sec", "checkpoint_complete", "evaluator_complete",
                "finite_training", "data_consistent", "code_consistent")
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("|" + "---|" * len(cols) + "\n")
        for row in rows:
            values = []
            for col in cols:
                value = row.get(col)
                if value is None:
                    rendered = ""
                elif isinstance(value, float):
                    rendered = f"{value:.4f}"
                else:
                    rendered = str(value)
                values.append(rendered)
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n## Pre-registered deltas\n\n```json\n")
        handle.write(json.dumps(deltas, indent=2) + "\n```\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
