from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_contract():
    return yaml.safe_load((ROOT / "contracts/exp15.yaml").read_text(encoding="utf-8"))


def test_handoff_contract_validator_passes():
    subprocess.run(
        [sys.executable, "scripts/exp15/validate_contract.py", "handoff"],
        cwd=ROOT,
        check=True,
    )


def test_preflight_has_valid_shell_syntax():
    subprocess.run(
        ["bash", "-n", "scripts/exp15/00_agent_preflight.sh"],
        cwd=ROOT,
        check=True,
    )


def test_resource_partition_and_arm_matrix_are_frozen():
    contract = load_contract()
    resources = contract["resources"]
    assert resources["workers"] == 24
    assert resources["gpus_per_worker"] == 4
    assert resources["arms"] == 6
    assert resources["workers_per_arm"] == 4
    assert resources["gpus_per_arm"] == 16
    assert resources["total_gpus"] == 96
    assert len(contract["arms"]) == 6
    assert {arm["seed"] for arm in contract["arms"]} == {0, 1}
    assert [arm["objectives"] for arm in contract["arms"]].count(["ce"]) == 2
    assert [arm["objectives"] for arm in contract["arms"]].count(["ce", "observation"]) == 2
    assert [arm["objectives"] for arm in contract["arms"]].count(
        ["ce", "observation", "event"]
    ) == 2


def test_scientific_contract_blocks_known_confounds():
    contract = load_contract()
    training = contract["training"]
    model = contract["model_contract"]
    assert training["pilot_optimizer_steps"] == 4000
    assert training["checkpoint_steps"] == [500, 1000, 2000, 4000]
    assert training["native_dynamic_visual_tokens"] is True
    assert training["manual_visual_token_k_sweep"] is False
    assert training["ce_exposure_identical_across_arms"] is True
    assert contract["data_contract"]["ce"]["temporal_qa_ratio"] == 0.0
    assert model["observation_path"]["queries"] == 256
    assert model["event_path"]["queries_per_query_set"] == 256
    assert model["transition_head"]["hidden_multiplier"] == 8
    assert model["visual_pair_loss"] == {"mse_weight": 0.1, "cosine_weight": 0.9}
    assert model["event_path"]["student_input_order"] == [
        "source frame",
        "Query1",
        "adjacent-event instruction",
        "Query2",
    ]


def test_server_runbook_names_all_hard_smoke_levels():
    text = (ROOT / "docs/EXP15_SERVER_AGENT.md").read_text(encoding="utf-8")
    for phrase in (
        "One GPU",
        "One 4-GPU Worker",
        "Two Workers / 8 GPUs",
        "24-Worker job dry-run",
        "Do not pull inside a GPU Pod",
        "source frame -> Query1 -> adjacent-event instruction -> Query2",
    ):
        assert phrase in text
