"""MVBench/TempCompass evaluation with the native Qwen3-VL processor and generation.

This is an external-validity anchor for EXP-12, not a replacement for the historical
paired evaluator.  It keeps the same 32 benchmark frames but restores Qwen's native
aspect-preserving resize, dynamic visual grid, per-temporal-unit delimiters/timestamps,
MRoPE construction, and greedy answer generation.  The cluster environment has no
torchvision, so the small preprocessing compatibility layer below implements the exact
Qwen3-VL smart-resize/patch-layout math using torch and the model's local processor
configuration instead of instantiating ``AutoProcessor``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from ..data.video_io import decode_frames, patchify
from .mcq_eval import load_image_frames, load_items
from .mcq_utils import generated_letter, parse_options, result_document, target_letter
from .native_checkpoint import apply_native_overlay


ANSWER_INSTRUCTION = "Select the best answer. Respond with only its option letter."
VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _duration_seconds(item: dict) -> float | None:
    meta = item.get("meta") or {}
    for container in (item, meta):
        for key in ("duration", "video_duration", "duration_seconds", "时长"):
            try:
                value = float(container.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def load_native_frames(
    item: dict,
    num_frames: int,
    fallback_fps: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Load the same pre-extracted frame source as the historical evaluator."""
    meta = item.get("meta") or {}
    images_info = meta.get("images_info") or []
    duration = _duration_seconds(item)
    if images_info:
        frames = load_image_frames(images_info, num_frames)
        effective_fps = (num_frames - 1) / duration if duration and num_frames > 1 else fallback_fps
        return frames, {
            "frame_source": "images_info",
            "timestamp_source": "duration" if duration else "fallback_fps",
            "timestamp_fps": effective_fps,
            "duration_seconds": duration or num_frames / fallback_fps,
        }

    rng = np.random.default_rng(seed)
    frames, diagnostics = decode_frames(
        item["视频"], num_frames, fallback_fps, "fps_or_uniform",
        random_offset=False, rng=rng, return_metadata=True,
    )
    effective_fps = float(diagnostics.get("effective_fps") or fallback_fps)
    return frames, {
        "frame_source": "video_pyav",
        "timestamp_source": "decoded_frame_ids",
        "timestamp_fps": effective_fps,
        "duration_seconds": (num_frames - 1) / effective_fps if num_frames > 1 else 0.0,
    }


def load_video_preprocess_config(model_path: str) -> dict:
    """Load and validate the official local Qwen video-processor contract."""
    path = os.path.join(model_path, "video_preprocessor_config.json")
    with open(path) as handle:
        config = json.load(handle)
    expected = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported native processor setting {key}={config.get(key)!r}")
    size = config.get("size") or {}
    if not size.get("shortest_edge") or not size.get("longest_edge"):
        raise ValueError(f"invalid video processor size in {path}")
    config["_path"] = os.path.abspath(path)
    return config


def native_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Torchvision-free equivalent of Qwen3-VL ``smart_resize``."""
    if height < factor or width < factor:
        raise ValueError(f"height={height} or width={width} is smaller than factor={factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    padded_frames = math.ceil(num_frames / temporal_factor) * temporal_factor
    if padded_frames * resized_height * resized_width > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif padded_frames * resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def native_preprocess_frames(frames: np.ndarray, processor_config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen smart resize, normalization and native patchification."""
    num_frames, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"expected RGB frames, got shape {frames.shape}")
    patch_size = int(processor_config["patch_size"])
    temporal_patch = int(processor_config["temporal_patch_size"])
    merge_size = int(processor_config["merge_size"])
    size = processor_config["size"]
    resized_height, resized_width = native_smart_resize(
        num_frames,
        height,
        width,
        temporal_factor=temporal_patch,
        factor=patch_size * merge_size,
        min_pixels=int(size["shortest_edge"]),
        max_pixels=int(size["longest_edge"]),
    )
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    return patchify(tensor, duplicate_frames=False, temporal_patch_size=temporal_patch)


def video_replacement(grid_thw: torch.Tensor, timestamp_fps: float) -> str:
    """Build Qwen3-VL's timestamp-separated per-unit visual placeholders."""
    grid_t, grid_h, grid_w = [int(value) for value in grid_thw.tolist()]
    frame_tokens = grid_h * grid_w // 4
    timestamps = [((2 * index) + 0.5) / timestamp_fps for index in range(grid_t)]
    return "".join(
        f"<{timestamp:.1f} seconds>{VISION_START_TOKEN}"
        f"{VIDEO_TOKEN * frame_tokens}{VISION_END_TOKEN}"
        for timestamp in timestamps
    )


def build_prompt(tokenizer, question: str) -> str:
    """Render one video/user turn while leaving replacement to the processor."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": f"{question.rstrip()}\n{ANSWER_INSTRUCTION}"},
        ],
    }]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def build_native_inputs(
    tokenizer,
    question: str,
    frames: np.ndarray,
    timestamp_fps: float,
    processor_config: dict,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build model inputs equivalent to Qwen3VLProcessor for one preloaded video."""
    pixel_values, grid = native_preprocess_frames(frames, processor_config)
    prompt = build_prompt(tokenizer, question)
    if prompt.count(VIDEO_TOKEN) != 1:
        raise ValueError("Qwen chat template must contain exactly one video placeholder")
    prompt = prompt.replace(VIDEO_TOKEN, video_replacement(grid, timestamp_fps), 1)
    tokenized = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
        return_token_type_ids=False,
    )
    video_token_id = tokenizer.convert_tokens_to_ids(VIDEO_TOKEN)
    mm_token_type_ids = torch.zeros_like(tokenized["input_ids"])
    mm_token_type_ids[tokenized["input_ids"] == video_token_id] = 2
    expected_video_tokens = int(grid.prod().item() // 4)
    actual_video_tokens = int((mm_token_type_ids == 2).sum().item())
    if actual_video_tokens != expected_video_tokens:
        raise ValueError(
            f"video placeholder mismatch: text={actual_video_tokens}, grid={expected_video_tokens}"
        )
    inputs = dict(tokenized)
    inputs.update({
        "pixel_values_videos": pixel_values,
        "video_grid_thw": grid.unsqueeze(0),
        "mm_token_type_ids": mm_token_type_ids,
    })
    return inputs, {
        "video_grid_thw": grid.tolist(),
        "native_video_tokens": expected_video_tokens,
        "resized_height": int(grid[1]) * 16,
        "resized_width": int(grid[2]) * 16,
    }


def _move_inputs(inputs, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local Qwen3-VL model/processor path")
    parser.add_argument("--overlay", default="", help="native-key EXP-12 overlay from native_checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="native_qwen_matched32_generation")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--timestamp-fps", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    if args.num_frames < 2 or args.num_frames % 2:
        parser.error("num_frames must be an even integer >= 2")
    if args.timestamp_fps <= 0:
        parser.error("timestamp_fps must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor_config = load_video_preprocess_config(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    overlay_audit = apply_native_overlay(model, args.overlay) if args.overlay else None
    model.requires_grad_(False).eval().to(args.device)

    all_items = load_items(args.data, args.task, args.max_clips)
    items = [
        (index, item) for index, item in enumerate(all_items)
        if index % args.num_shards == args.shard_index
    ]
    print(
        f"{args.task}: {len(items)}/{len(all_items)} native-generation items "
        f"(shard {args.shard_index}/{args.num_shards})"
    )
    records: list[dict] = []
    skipped = 0
    token_counts: list[int] = []
    for local_index, (item_index, item) in enumerate(items):
        question = str(item.get("问题", ""))
        options = parse_options(question)
        gold = target_letter(str(item.get("目标值", "")))
        valid_letters = [letter for letter, _ in options]
        if len(options) < 2 or gold is None or gold not in valid_letters:
            skipped += 1
            continue
        try:
            frames, frame_audit = load_native_frames(
                item,
                args.num_frames,
                args.timestamp_fps,
                args.seed * 100003 + item_index,
            )
            inputs, native_audit = build_native_inputs(
                tokenizer,
                question,
                frames,
                float(frame_audit["timestamp_fps"]),
                processor_config,
            )
            grid = native_audit["video_grid_thw"]
            native_tokens = native_audit["native_video_tokens"]
            inputs = _move_inputs(inputs, args.device)
            prompt_length = int(inputs["input_ids"].shape[1])
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            answer = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )[0].strip()
            pred = generated_letter(answer, valid_letters)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{item_index}] native eval failed: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        token_counts.append(native_tokens)
        records.append({
            "idx": item_index,
            "pred": pred,
            "gold": gold,
            "sub_type": item.get("子类别", "?"),
            "ok": int(pred == gold),
            "generated_text": answer,
            "native_video_tokens": native_tokens,
            "video_grid_thw": grid,
            "original_height": int(frames.shape[1]),
            "original_width": int(frames.shape[2]),
            "resized_height": native_audit["resized_height"],
            "resized_width": native_audit["resized_width"],
            **frame_audit,
        })
        if (local_index + 1) % 50 == 0:
            correct = sum(record["ok"] for record in records)
            print(
                f"{local_index + 1}/{len(items)} acc="
                f"{100 * correct / max(len(records), 1):.2f}%"
            )

    metadata = {
        "model": os.path.abspath(args.model),
        "overlay": overlay_audit,
        "frame_protocol": "matched_preextracted_32_native_compatible_smart_resize",
        "num_frames": args.num_frames,
        "fallback_timestamp_fps": args.timestamp_fps,
        "answer_instruction": ANSWER_INSTRUCTION,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "processor_implementation": "torchvision_free_qwen3vl_compat_v1",
        "video_preprocessor_config": processor_config["_path"],
        "evaluator_commit": _git_commit(),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_clips": args.max_clips,
    }
    if token_counts:
        metadata["native_video_tokens"] = {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
        }
    document = result_document(
        task=args.task,
        protocol=args.protocol,
        scoring="native_qwen_greedy_generation:option_letter",
        records=records,
        skipped=skipped,
        metadata=metadata,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(document, handle, ensure_ascii=False)
    print(
        f"{args.task}: {document['correct']}/{document['total']} "
        f"= {100 * document['acc']:.2f}%; skipped={skipped}; output={args.output}"
    )


if __name__ == "__main__":
    main()
