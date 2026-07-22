import json
import subprocess
import sys

import yaml

from jepa_vlm.config import load_config, resolved_raw_num_frames, resolved_visual_tokens


ARMS = {
    "b0_ce_seed1": (1, "none", 0.0),
    "b1_query_seed1": (1, "query", 0.0),
    "b2_noquery_seed0": (0, "no_query", 0.0),
    "b3_noquery_seed1": (1, "no_query", 0.0),
    "b4_query_beatcopy_seed0": (0, "query", 1.0),
    "b5_query_beatcopy_seed1": (1, "query", 1.0),
}


def _evaluation(values, task):
    records = [
        {"idx": index, "gold": "A", "pred": "A" if ok else "B", "ok": int(ok),
         "sub_type": "order"}
        for index, ok in enumerate(values)
    ]
    return {"task": task, "acc": sum(values) / len(values), "correct": sum(values),
            "total": len(values), "skipped": 0, "results": records}


def _arm_artifacts(root, arm, predictive):
    arm_root = root / arm
    checkpoint = arm_root / "checkpoint-800"
    checkpoint.mkdir(parents=True)
    state = checkpoint / "state.pt"
    state.write_bytes(b"complete")
    (checkpoint / "checkpoint_meta.json").write_text(json.dumps({
        "step": 800, "step_unit": "optimizer_update", "state_bytes": len(b"complete")
    }))
    log = {"step": 800, "loss": 1.0, "ce_loss": 1.0, "max_memory_gb": 1.0,
           "samples_per_sec": 2.0}
    if predictive:
        log.update({"state/centered_margin": 0.2, "state/persistence_ratio": 0.8,
                    "state/beat_copy_loss": 0.1, "state/retrieval_top1": 0.2,
                    "state/retrieval_top5": 0.5})
    (arm_root / "trainer_log.jsonl").write_text(json.dumps(log) + "\n")
    (arm_root / "manifest.sha256").write_text("a" * 64 + "\n")
    (arm_root / "git_commit.txt").write_text("1" * 40 + "\n")
    for task, filename in (("MVBench", "mvbench"), ("Tempcompass", "tempcompass")):
        (arm_root / f"checkpoint-800_{filename}.json").write_text(
            json.dumps(_evaluation([1, 1, 0, 0], task))
        )


def test_exp14_configs_freeze_everything_except_registered_factors():
    for arm, (seed, mode, beat_weight) in ARMS.items():
        cfg = load_config(f"configs/exp14_state_diagnostics/{arm}.yaml")
        assert resolved_visual_tokens(cfg) == 64
        assert resolved_raw_num_frames(cfg) == 32
        assert cfg.train.max_steps == 800
        assert cfg.train.save_every == 400
        assert cfg.train.seed == seed
        assert cfg.model.state_predictor_mode == mode
        assert cfg.model.beat_copy_loss_weight == beat_weight
        assert cfg.model.state_loss_weight == 0.05
        assert cfg.train.train_vision is False


def test_exp14_job_partitions_six_eight_gpu_worlds():
    with open("job_exp14.yaml") as handle:
        job = yaml.safe_load(handle)
    assert job["spec"]["Worker"]["num"] == 12
    assert job["spec"]["Worker"]["limits"]["gpu"] == "4"
    command = job["run"]["command"]
    assert "EXP14_NODES_PER_ARM=2" in command
    assert "EXP14_GRAD_ACCUM=4" in command
    entry = open("scripts/cluster/job_exp14_entry.sh").read()
    assert "GROUP_ID" in entry
    assert "EFFECTIVE_BATCH" in entry
    assert 'eval400_${ARM}' in entry and 'eval800_${ARM}' in entry


def test_exp14_collector_joins_frozen_k64_controls(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "new"
    for arm in ("a4_ce_k64", "a5_query_k64"):
        _arm_artifacts(source, arm, arm == "a5_query_k64")
    for arm, (_, mode, _) in ARMS.items():
        _arm_artifacts(root, arm, mode != "none")
    subprocess.run(
        [sys.executable, "scripts/exp14/02_collect_results.py", "--root", str(root),
         "--source", str(source)],
        check=True, stdout=subprocess.DEVNULL,
    )
    result = json.loads((root / "comparison.json").read_text())
    assert result["complete"] is True
    assert len(result["rows"]) == 8
    assert result["decision"]["auto_event_submission"] is False
    assert "seed0_beatcopy_minus_query:MVBench" in result["paired_tests"]


def test_official_budget_collector_labels_reproduction(tmp_path):
    protocols = (
        "official_budget_base_full_generation",
        "official_budget_ckpt_k64_full_generation",
        "official_budget_base_cap32_generation",
        "official_budget_ckpt_k64_cap32_generation",
    )
    for index, protocol in enumerate(protocols):
        document = _evaluation([1, 1, index % 2, 0], "MVBench")
        document["protocol"] = protocol
        document["metadata"] = {"native_video_tokens": {"median": 1024}}
        (tmp_path / f"{protocol}_mvbench.json").write_text(json.dumps(document))
    subprocess.run(
        [sys.executable, "scripts/exp13/02_collect_official.py", "--root", str(tmp_path)],
        check=True, stdout=subprocess.DEVNULL,
    )
    result = json.loads((tmp_path / "official_budget_comparison.json").read_text())
    assert "official-budget reproduction" in result["label"]
    assert len(result["rows"]) == 4
    assert "frame_budget_effect_base" in result["paired_tests"]
