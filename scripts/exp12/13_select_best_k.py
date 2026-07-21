#!/usr/bin/env python3
"""Apply the pre-registered EXP-12 K promotion gates."""

from __future__ import annotations

import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    args = parser.parse_args()
    doc = json.load(open(args.comparison))
    root = os.path.dirname(os.path.abspath(args.comparison))
    rows = {(row["K"], row["mode"]): row for row in doc["rows"]}
    candidates = []
    audits = []
    for k in (4, 16, 64):
        ce, query = rows[(k, "none")], rows[(k, "query")]
        tc_delta = query["TempCompass"] - ce["TempCompass"]
        mv_delta = query["MVBench"] - ce["MVBench"]
        temporal_delta = sum(
            (query.get(key) or 0) - (ce.get(key) or 0) for key in ("order", "direction")
        )
        rank_slope = query.get("target_effective_rank_slope")
        gates = {
            "tempcompass_not_clearly_negative": tc_delta >= -0.3,
            "mvbench_drop_within_0.3": mv_delta >= -0.3,
            "centered_margin_gt_0.10": (query.get("centered_margin") or -999) > 0.10,
            "persistence_ratio_lt_0.90": (query.get("persistence_ratio") or 999) < 0.90,
            "effective_rank_not_declining": (
                rank_slope is not None and math.isfinite(float(rank_slope))
                and float(rank_slope) >= -0.01
            ),
            "complete_and_finite": bool(
                query.get("checkpoint_complete")
                and query.get("evaluator_complete")
                and query.get("finite_training")
                and query.get("data_consistent")
                and query.get("code_consistent")
            ),
        }
        passed = all(gates.values())
        item = {"K": k, "passed": passed, "gates": gates, "tempcompass_delta": tc_delta,
                "mvbench_delta": mv_delta, "order_direction_delta": temporal_delta,
                "persistence_ratio": query.get("persistence_ratio"),
                "samples_per_sec": query.get("samples_per_sec"),
                "max_memory_gb": query.get("max_memory_gb")}
        audits.append(item)
        if passed:
            candidates.append(item)
    selection = None
    if candidates:
        candidates.sort(key=lambda row: (
            row["tempcompass_delta"], row["order_direction_delta"], row["mvbench_delta"],
            -(row["persistence_ratio"] or 999), row["samples_per_sec"] or 0,
        ), reverse=True)
        selection = candidates[0]
        k16 = next((row for row in candidates if row["K"] == 16), None)
        if selection["K"] == 64 and k16:
            gain = rows[(64, "query")]["TempCompass"] - rows[(16, "query")]["TempCompass"]
            if gain < 0.3:
                selection = k16
                selection = {**selection, "efficiency_override": "K64 gain over K16 < 0.3"}
    result = {"status": "PASS" if selection else "FAIL", "best_k": selection["K"] if selection else None,
              "selection": selection, "candidate_audit": audits,
              "event_auto_submit": False}
    if not selection:
        result["recommendation"] = "Do not start Event automatically; prioritize no-query or beat-copy evidence."
    json.dump(result, open(os.path.join(root, "selection.json"), "w"), indent=2)
    with open(os.path.join(root, "selection.md"), "w") as handle:
        handle.write(f"# EXP-12 K selection: {result['status']}\n\n")
        handle.write(f"Best K: {result['best_k']}\n\n")
        handle.write("Event experiments were not auto-submitted.\n\n```json\n")
        handle.write(json.dumps(result, indent=2) + "\n```\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
