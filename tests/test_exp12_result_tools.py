import json
import subprocess
import sys


ARMS = (
    "a0_ce_k4", "a1_query_k4", "a2_ce_k16",
    "a3_query_k16", "a4_ce_k64", "a5_query_k64",
)


def _evaluation(ok_values, task):
    results = [
        {
            "idx": index,
            "pred": "A" if ok else "B",
            "gold": "A",
            "sub_type": "order" if index % 2 else "direction",
            "ok": int(ok),
            "option_scores": {"A": 0.1, "B": 0.2},
        }
        for index, ok in enumerate(ok_values)
    ]
    return {
        "task": task,
        "acc": sum(ok_values) / len(ok_values),
        "correct": sum(ok_values),
        "total": len(ok_values),
        "skipped": 0,
        "results": results,
    }


def test_collect_and_select_exp12_results(tmp_path):
    root = tmp_path / "results"
    commit = "1" * 40
    for index, arm in enumerate(ARMS):
        arm_root = root / arm
        checkpoint = arm_root / "checkpoint-800"
        checkpoint.mkdir(parents=True)
        state = checkpoint / "state.pt"
        state.write_bytes(b"complete")
        (checkpoint / "checkpoint_meta.json").write_text(json.dumps({
            "step": 800,
            "step_unit": "optimizer_update",
            "state_bytes": state.stat().st_size,
        }))
        (arm_root / "manifest.sha256").write_text("a" * 64 + "\n")
        (arm_root / "git_commit.txt").write_text(commit + "\n")
        query = index % 2 == 1
        log = {
            "step": 800,
            "interval_steps": 10,
            "sec_per_step": 1.0,
            "loss": 1.0,
            "ce_loss": 1.0,
            "samples_per_sec": 32.0,
            "max_memory_gb": 10.0,
        }
        if query:
            log.update({
                "state/centered_margin": 0.2,
                "state/persistence_ratio": 0.8,
                "state/dynamic_sample_fraction": 0.5,
                "state/target_effective_rank": 8.0,
                "state/retrieval_top1": 0.2,
                "state/retrieval_top5": 0.5,
            })
        (arm_root / "trainer_log.jsonl").write_text(json.dumps(log) + "\n")
        ok = [1, 1, 1, 0] if query else [1, 1, 0, 0]
        (arm_root / "checkpoint-800_mvbench.json").write_text(
            json.dumps(_evaluation(ok, "MVBench"))
        )
        (arm_root / "checkpoint-800_tempcompass.json").write_text(
            json.dumps(_evaluation(ok, "Tempcompass"))
        )

    subprocess.run(
        [sys.executable, "scripts/exp12/12_collect_results.py", "--root", str(root)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    comparison = json.loads((root / "comparison.json").read_text())
    assert len(comparison["rows"]) == 6
    assert comparison["deltas"]["query_minus_ce_k16"]["TempCompass"] == 25.0
    assert all(row["data_consistent"] for row in comparison["rows"])
    assert all(row["code_consistent"] for row in comparison["rows"])

    subprocess.run(
        [
            sys.executable,
            "scripts/exp12/13_select_best_k.py",
            "--comparison", str(root / "comparison.json"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    selection = json.loads((root / "selection.json").read_text())
    assert selection["status"] == "PASS"
    assert selection["best_k"] in (4, 16, 64)
    assert selection["event_auto_submit"] is False
