import json
import subprocess
import sys

import numpy as np
import torch
import yaml

from jepa_vlm.probes.mcq_utils import (
    generated_letter,
    parse_options,
    result_document,
    target_letter,
)
from jepa_vlm.probes.merge_mcq_results import merge_documents
from jepa_vlm.probes.native_checkpoint import (
    export_native_overlay,
    map_wrapper_state_to_native,
)
from jepa_vlm.probes.native_qwen_mcq_eval import (
    ANSWER_INSTRUCTION,
    OFFICIAL_MAX_FRAMES,
    OFFICIAL_MAX_TOKENS_PER_UNIT,
    OFFICIAL_MAX_TOTAL_VIDEO_TOKENS,
    OFFICIAL_MVBENCH_INSTRUCTION,
    VIDEO_TOKEN,
    build_prompt,
    native_preprocess_frames,
    native_smart_resize,
    official_frame_count,
    official_max_pixels,
    official_mvbench_text,
    video_replacement,
)


def _records(values, offset=0):
    return [
        {
            "idx": offset + index,
            "pred": "A" if ok else "B",
            "gold": "A",
            "sub_type": "order" if index % 2 else "direction",
            "ok": int(ok),
        }
        for index, ok in enumerate(values)
    ]


def _document(protocol, task, values, offset=0, scoring="test"):
    return result_document(
        task=task,
        protocol=protocol,
        scoring=scoring,
        records=_records(values, offset),
        skipped=0,
        metadata={"max_clips": 0},
    )


def test_mcq_parsing_and_generated_letter():
    question = "What changes?\n(A) opens\nB. closes\nC: stays still"
    assert [letter for letter, _ in parse_options(question)] == ["A", "B", "C"]
    assert target_letter("(B) closes") == "B"
    assert generated_letter("B", {"A", "B", "C"}) == "B"
    assert generated_letter("The answer is (C).", {"A", "B", "C"}) == "C"
    assert generated_letter("A person opens the door", {"A", "B", "C"}) is None
    assert generated_letter("D", {"A", "B", "C"}) is None


def test_result_schema_and_shard_merge():
    left = _document("native", "MVBench", [1, 0], offset=0)
    right = _document("native", "MVBench", [1, 1], offset=2)
    left["results"][0]["native_video_tokens"] = 512
    right["results"][0]["native_video_tokens"] = 1024
    merged = merge_documents([right, left])
    assert [row["idx"] for row in merged["results"]] == [0, 1, 2, 3]
    assert merged["correct"] == 3
    assert merged["categories"]["direction"]["total"] == 2
    assert merged["metadata"]["native_video_tokens"]["median"] == 768


def test_duplicate_shards_are_rejected():
    document = _document("native", "MVBench", [1])
    try:
        merge_documents([document, document])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate indices must fail")


def test_checkpoint_mapping_and_compact_export(tmp_path):
    wrapper_state = {
        "language_model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "visual.patch_embed.proj.weight": torch.ones(1),
        "state_transition_head.fc1.weight": torch.zeros(1),
    }
    mapped, ignored = map_wrapper_state_to_native(wrapper_state)
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in mapped
    assert "model.visual.patch_embed.proj.weight" in mapped
    assert ignored == ["state_transition_head.fc1.weight"]

    checkpoint = tmp_path / "checkpoint-800"
    checkpoint.mkdir()
    torch.save({
        "model": wrapper_state,
        "optimizer": {"large_unused_state": torch.zeros(3)},
        "step": 800,
    }, checkpoint / "state.pt")
    output = tmp_path / "overlay.pt"
    metadata = export_native_overlay(str(checkpoint), str(output))
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert metadata["source_step"] == 800
    assert len(payload["model"]) == 2
    assert "optimizer" not in payload


def test_native_prompt_uses_video_turn_and_fixed_answer_instruction():
    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation[0]["content"][0] == {"type": "video"}
            assert ANSWER_INSTRUCTION in conversation[0]["content"][1]["text"]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "rendered"

    assert build_prompt(Processor(), "question") == "rendered"


def test_official_budget_prompt_and_token_math():
    question = "What happens next?\n(A) opens\nB. closes\nC: stays still\nD) disappears"
    text = official_mvbench_text(question)
    assert text.startswith(OFFICIAL_MVBENCH_INSTRUCTION)
    assert "Question: What happens next? Possible answer choices:" in text
    assert text.endswith("The best answer is:")

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation[0]["content"][1]["text"] == text
            return "official-rendered"

    assert build_prompt(Tokenizer(), question, "official_mvbench") == "official-rendered"
    assert official_frame_count(8.0) == 16
    assert official_frame_count(10_000.0) == OFFICIAL_MAX_FRAMES
    assert official_max_pixels(32) == 32 * OFFICIAL_MAX_TOKENS_PER_UNIT * 1024
    assert official_max_pixels(OFFICIAL_MAX_FRAMES) == OFFICIAL_MAX_TOTAL_VIDEO_TOKENS * 2048


def test_torchvision_free_native_preprocess_and_timestamps():
    assert native_smart_resize(
        32, 64, 96,
        temporal_factor=2,
        factor=32,
        min_pixels=4096,
        max_pixels=25165824,
    ) == (64, 96)
    config = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "size": {"shortest_edge": 4096, "longest_edge": 25165824},
    }
    pixels, grid = native_preprocess_frames(
        np.zeros((32, 64, 96, 3), dtype=np.uint8), config
    )
    assert grid.tolist() == [16, 4, 6]
    assert pixels.shape == (16 * 4 * 6, 3 * 2 * 16 * 16)
    replacement = video_replacement(grid, 4.0)
    assert replacement.count(" seconds>") == 16
    assert replacement.count(VIDEO_TOKEN) == 16 * 4 * 6 // 4
    assert replacement.startswith("<0.1 seconds><|vision_start|>")


def test_native_anchor_collector_builds_paired_matrix(tmp_path):
    protocols = (
        "custom_base_k4_full_option",
        "custom_base_k16_full_option",
        "custom_base_k64_full_option",
        "custom_ckpt_k64_full_option",
        "custom_base_k64_letter",
        "custom_ckpt_k64_letter",
        "native_base_matched32_generation",
        "native_ckpt_k64_matched32_generation",
    )
    for protocol_index, protocol in enumerate(protocols):
        for task in ("MVBench", "Tempcompass"):
            document = _document(protocol, task, [1, protocol_index % 2, 0, 1])
            (tmp_path / f"{protocol}_{task.lower()}.json").write_text(json.dumps(document))
    subprocess.run(
        [
            sys.executable,
            "scripts/exp12/18_collect_native_anchor.py",
            "--root", str(tmp_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    comparison = json.loads((tmp_path / "native_anchor_comparison.json").read_text())
    assert comparison["complete"] is True
    assert len(comparison["rows"]) == 16
    assert "native_training_effect_k64:MVBench" in comparison["paired_tests"]


def test_exp13_company_job_uses_seven_evaluation_workers():
    with open("job_exp13_eval.yaml") as handle:
        job = yaml.safe_load(handle)
    assert job["spec"]["Worker"]["num"] == 7
    assert job["spec"]["Worker"]["limits"]["gpu"] == "4"
    command = job["run"]["command"]
    assert "job_exp13_eval_entry.sh" in command
    assert "EXP13_ATTEMPT_ID=unset" in command
    assert "EXP13_MAX_CLIPS=0" in command


def test_exp13_official_budget_job_uses_four_evaluation_workers():
    with open("job_exp13_official.yaml") as handle:
        job = yaml.safe_load(handle)
    assert job["spec"]["Worker"]["num"] == 4
    assert job["spec"]["Worker"]["limits"]["gpu"] == "4"
    command = job["run"]["command"]
    assert "job_exp13_official_entry.sh" in command
    assert "EXP13_OFFICIAL_MAX_CLIPS=0" in command
    assert "flash_attention_2" in command
