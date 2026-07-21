from pathlib import Path
import subprocess
import sys


def test_exp12_job_partitions_six_independent_worlds():
    path = Path("scripts/cluster/job_exp12_entry.sh")
    assert path.exists()
    text = path.read_text()
    assert "GROUP_ID" in text
    assert "NODES_PER_ARM" in text
    assert "EXPECTED_NNODES" in text
    assert "ARMS" in text
    assert "WORLD_SIZE=24" not in text
    assert 'eval400_${ARM}' in text
    assert '${ARM}.evaldone' in text


def test_event_config_materializer_fills_only_k(tmp_path):
    output = tmp_path / "event"
    subprocess.run(
        [
            sys.executable,
            "scripts/exp12/materialize_event_configs.py",
            "--best-k", "16", "--output", str(output),
        ],
        check=True,
    )
    configs = sorted(output.glob("b*.yaml"))
    assert len(configs) == 6
    assert all("PLACEHOLDER_K" not in path.read_text() for path in configs)
    assert all("visual_tokens_per_unit: 16" in path.read_text() for path in configs)
