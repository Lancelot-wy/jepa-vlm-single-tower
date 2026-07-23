#!/usr/bin/env python3
"""Validate the EXP-15 handoff, implementation, or launch contract.

The handoff stage is expected to pass on this commit. The implementation and
launch stages intentionally fail until the server Agent creates and validates
the artifacts listed in contracts/exp15.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "exp15.yaml"


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def read_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("contract root must be a mapping")
    return data


def add(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_handoff(contract: dict[str, Any], errors: list[str]) -> None:
    required = [
        "CLAUDE.md",
        ".claude/commands/exp15.md",
        "docs/EXP15_SERVER_AGENT.md",
        "contracts/exp15.yaml",
        "results/exp15/AGENT_STATUS.md",
        "scripts/exp15/00_agent_preflight.sh",
        "scripts/exp15/validate_contract.py",
    ]
    for relative in required:
        path = ROOT / relative
        add(path.is_file() and path.stat().st_size > 0, f"missing handoff: {relative}", errors)

    experiment = contract.get("experiment", {})
    resources = contract.get("resources", {})
    arms = contract.get("arms", [])
    training = contract.get("training", {})
    add(experiment.get("id") == "EXP-15", "experiment id must be EXP-15", errors)
    add(
        experiment.get("branch") == "exp15-native-orca",
        "contract branch must be exp15-native-orca",
        errors,
    )
    add(resources.get("workers") == 24, "resources.workers must be 24", errors)
    add(resources.get("gpus_per_worker") == 4, "gpus_per_worker must be 4", errors)
    add(resources.get("total_gpus") == 96, "total_gpus must be 96", errors)
    add(resources.get("arms") == 6, "resources.arms must be 6", errors)
    add(resources.get("workers_per_arm") == 4, "workers_per_arm must be 4", errors)
    add(len(arms) == 6, "exactly six arms are required", errors)
    arm_ids = [arm.get("id") for arm in arms]
    add(len(set(arm_ids)) == 6, "arm ids must be unique", errors)
    add(
        {arm.get("seed") for arm in arms} == {0, 1},
        "both seed 0 and seed 1 are required",
        errors,
    )
    add(training.get("pilot_optimizer_steps") == 4000, "pilot must be 4000 steps", errors)
    add(
        training.get("checkpoint_steps") == [500, 1000, 2000, 4000],
        "checkpoint schedule mismatch",
        errors,
    )
    add(training.get("native_dynamic_visual_tokens") is True, "native tokens required", errors)
    add(training.get("manual_visual_token_k_sweep") is False, "K sweep must be disabled", errors)

    model = contract.get("model_contract", {})
    add(model.get("observation_path", {}).get("queries") == 256, "Observation needs 256 queries", errors)
    add(
        model.get("transition_head", {}).get("hidden_multiplier") == 8,
        "transition head must be D -> 8D -> D",
        errors,
    )
    pair_loss = model.get("visual_pair_loss", {})
    add(pair_loss.get("mse_weight") == 0.1, "pair MSE weight must be 0.1", errors)
    add(pair_loss.get("cosine_weight") == 0.9, "pair cosine weight must be 0.9", errors)


def validate_implementation(contract: dict[str, Any], errors: list[str]) -> None:
    artifacts = contract.get("required_artifacts_after_implementation", [])
    add(bool(artifacts), "required implementation artifact list is empty", errors)
    for relative in artifacts:
        path = ROOT / relative
        add(path.is_file() and path.stat().st_size > 0, f"missing implementation artifact: {relative}", errors)
        if path.is_file() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in ("PLACEHOLDER_K", "TODO_EXP15", "NOT_IMPLEMENTED_EXP15"):
                add(marker not in text, f"{relative} still contains {marker}", errors)

    for arm in contract.get("arms", []):
        config_path = ROOT / "configs" / "exp15" / f"{arm['id']}.yaml"
        if not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            errors.append(f"invalid YAML {config_path.relative_to(ROOT)}: {exc}")
            continue
        train = config.get("train", {})
        add(train.get("seed") == arm["seed"], f"seed mismatch in {config_path.name}", errors)
        add(bool(train.get("output_dir")), f"output_dir missing in {config_path.name}", errors)


def validate_launch(contract: dict[str, Any], errors: list[str]) -> None:
    try:
        branch = git("branch", "--show-current")
        status = git("status", "--porcelain")
        commit = git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        errors.append(f"git inspection failed: {exc}")
        return
    add(branch == contract["experiment"]["branch"], f"wrong branch: {branch}", errors)
    add(not status, "launch checkout must be clean", errors)
    add(len(commit) == 40, "launch commit must be a full hash", errors)

    job = ROOT / "job_exp15.yaml"
    entry = ROOT / "scripts" / "cluster" / "job_exp15_entry.sh"
    submit = ROOT / "scripts" / "exp15" / "03_submit.sh"
    if job.is_file():
        text = job.read_text(encoding="utf-8")
        add("num: 24" in text, "job_exp15.yaml must request 24 Workers", errors)
        add("restartPolicy: Never" in text, "restartPolicy must be Never", errors)
        add("EXP15_GIT_COMMIT" in text, "job must carry a fixed commit", errors)
    if entry.is_file():
        text = entry.read_text(encoding="utf-8")
        for token in ("GROUP_ID", "NODES_PER_ARM", "EXPECTED_NNODES", "TF_CONFIG"):
            add(token in text, f"job entry missing topology token {token}", errors)
        add("git pull" not in text, "GPU entrypoint must never git pull", errors)
    if submit.is_file():
        add(os.access(submit, os.X_OK), "03_submit.sh must be executable", errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("handoff", "implementation", "launch"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        contract = read_contract()
    except Exception as exc:  # clear one-line contract failure
        contract = {}
        errors.append(f"cannot read contract: {exc}")

    if contract:
        validate_handoff(contract, errors)
        if args.stage in {"implementation", "launch"}:
            validate_implementation(contract, errors)
        if args.stage == "launch":
            validate_launch(contract, errors)

    report = {
        "stage": args.stage,
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "root": str(ROOT),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
