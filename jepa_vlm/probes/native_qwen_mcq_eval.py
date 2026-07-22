"""MVBench/TempCompass evaluation with native-compatible Qwen3-VL generation.

This is an external-validity anchor for EXP-12, not a replacement for the historical
paired evaluator.  It keeps the same 32 benchmark frames but restores Qwen's native
aspect-preserving resize, dynamic visual grid, per-temporal-unit delimiters/timestamps,
MRoPE construction, and greedy answer generation.  The cluster environment has no
torchvision, so the small preprocessing compatibility layer below implements the exact
Qwen3-VL smart-resize/patch-layout math using torch and the model's local processor
configuration instead of instantiating ``AutoProcessor``.

The default remains the EXP-13 matched-32 diagnostic.  ``official_2fps`` is a
separate, explicitly labelled reproduction path: it decodes the real video at
2 fps (up to 2048 frames), applies the public 224K-total/640-per-unit visual
token budgets, and uses the technical-report MVBench prompt.  It is an
official-*budget* anchor, not a claim that this local HF runner is byte-for-byte
identical to Qwen's private evaluation service.
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
OFFICIAL_MVBENCH_INSTRUCTION = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
OFFICIAL_SAMPLE_FPS = 2.0
OFFICIAL_MAX_FRAMES = 2048
OFFICIAL_MAX_TOTAL_VIDEO_TOKENS = 224_000
OFFICIAL_MAX_TOKENS_PER_UNIT = 640


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


def _number(container: dict, keys: tuple[str, ...]) -> float | None:
    if not isinstance(container, dict):
        return None
    for key in keys:
        try:
            value = float(container.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _requested_segment(item: dict) -> tuple[float | None, float | None]:
    """Read common segment keys without assuming that every file is untrimmed."""
    meta = item.get("meta") or {}
    start_keys = ("start", "start_time", "start_seconds", "开始时间")
    end_keys = ("end", "end_time", "end_seconds", "结束时间")
    for container in (item, meta):
        start = _number(container, start_keys)
        end = _number(container, end_keys)
        if start is not None or end is not None:
            return start, end
    return None, None


def _video_timing(path: str) -> dict:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or 24.0)
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None and container.duration:
            duration = float(container.duration / av.time_base)
        if duration is None and stream.frames:
            duration = float(stream.frames) / native_fps
    if duration is None or duration <= 0:
        raise RuntimeError(f"cannot determine video duration: {path}")
    return {"native_fps": native_fps, "video_duration_seconds": duration}


def official_frame_count(
    duration_seconds: float,
    sample_fps: float = OFFICIAL_SAMPLE_FPS,
    max_frames: int = OFFICIAL_MAX_FRAMES,
    temporal_patch_size: int = 2,
) -> int:
    """Return a deterministic even frame count for Qwen's 2-fps policy."""
    if duration_seconds <= 0 or sample_fps <= 0:
        raise ValueError("duration_seconds and sample_fps must be positive")
    if max_frames < 4 or max_frames % temporal_patch_size:
        raise ValueError("max_frames must be >=4 and divisible by temporal_patch_size")
    desired = max(4, int(math.floor(duration_seconds * sample_fps)))
    desired = min(desired, max_frames)
    desired -= desired % temporal_patch_size
    return max(temporal_patch_size, desired)


def load_official_frames(
    item: dict,
    *,
    sample_fps: float = OFFICIAL_SAMPLE_FPS,
    max_frames: int = OFFICIAL_MAX_FRAMES,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Decode the actual benchmark video using the official-budget frame policy."""
    path = str(item.get("视频") or "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"official_2fps requires an accessible 视频 path: {path!r}")
    timing = _video_timing(path)
    video_duration = float(timing["video_duration_seconds"])
    requested_start, requested_end = _requested_segment(item)
    start = max(0.0, requested_start or 0.0)
    end = min(video_duration, requested_end) if requested_end is not None else video_duration
    segment_fallback = False
    # Some benchmark exports point at an already-trimmed clip while retaining
    # original-video bounds.  In that case the only valid policy is the whole file.
    if start >= video_duration or end <= start:
        start, end = 0.0, video_duration
        segment_fallback = True
    duration = end - start
    count = official_frame_count(duration, sample_fps, max_frames)
    rng = np.random.default_rng(seed)
    frames, diagnostics = decode_frames(
        path,
        count,
        sample_fps,
        "uniform",
        start=start,
        end=end,
        random_offset=False,
        rng=rng,
        return_metadata=True,
    )
    effective_fps = (count - 1) / duration if count > 1 and duration > 0 else sample_fps
    return frames, {
        "frame_source": "video_pyav",
        "frame_policy": "official_2fps_uniform_full_segment",
        "timestamp_source": "decoded_segment_duration",
        "timestamp_fps": effective_fps,
        "requested_sample_fps": sample_fps,
        "duration_seconds": duration,
        "video_duration_seconds": video_duration,
        "segment_start_seconds": start,
        "segment_end_seconds": end,
        "segment_bounds_fallback_to_trimmed_file": segment_fallback,
        "max_frames": max_frames,
        **diagnostics,
    }


def official_max_pixels(
    num_frames: int,
    *,
    temporal_patch_size: int = 2,
    max_total_video_tokens: int = OFFICIAL_MAX_TOTAL_VIDEO_TOKENS,
    max_tokens_per_unit: int = OFFICIAL_MAX_TOKENS_PER_UNIT,
) -> int:
    """Translate public Qwen visual-token limits into smart-resize pixels.

    For patch=16, temporal_patch=2 and spatial merge=2, one final video token
    represents 2048 spatiotemporal pixels; one per-unit token represents 1024
    spatial pixels.  The tighter of the total and per-unit budgets wins.
    """
    if num_frames < 1 or temporal_patch_size < 1:
        raise ValueError("num_frames and temporal_patch_size must be positive")
    if max_total_video_tokens < 1 or max_tokens_per_unit < 1:
        raise ValueError("visual-token budgets must be positive")
    padded_frames = math.ceil(num_frames / temporal_patch_size) * temporal_patch_size
    total_budget = max_total_video_tokens * 2048
    per_unit_budget = padded_frames * max_tokens_per_unit * 1024
    return min(total_budget, per_unit_budget)


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


def native_preprocess_frames(
    frames: np.ndarray,
    processor_config: dict,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen smart resize, normalization and native patchification."""
    num_frames, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"expected RGB frames, got shape {frames.shape}")
    patch_size = int(processor_config["patch_size"])
    temporal_patch = int(processor_config["temporal_patch_size"])
    merge_size = int(processor_config["merge_size"])
    size = processor_config["size"]
    resolved_min_pixels = int(size["shortest_edge"] if min_pixels is None else min_pixels)
    resolved_max_pixels = int(size["longest_edge"] if max_pixels is None else max_pixels)
    if resolved_max_pixels < resolved_min_pixels:
        raise ValueError(
            f"max_pixels={resolved_max_pixels} is smaller than min_pixels={resolved_min_pixels}"
        )
    resized_height, resized_width = native_smart_resize(
        num_frames,
        height,
        width,
        temporal_factor=temporal_patch,
        factor=patch_size * merge_size,
        min_pixels=resolved_min_pixels,
        max_pixels=resolved_max_pixels,
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


def official_mvbench_text(question: str) -> str:
    """Render the public Qwen technical-report MVBench question format."""
    options = parse_options(question)
    option_lines = {line for _, line in options}
    stem_lines = [line.strip() for line in question.splitlines() if line.strip() not in option_lines]
    stem = "\n".join(line for line in stem_lines if line).strip()
    choices = "\n".join(line for _, line in options)
    if not stem or len(options) < 2:
        raise ValueError("official MVBench prompt requires a question stem and >=2 options")
    return (
        f"{OFFICIAL_MVBENCH_INSTRUCTION}\n"
        f"Question: {stem} Possible answer choices:\n"
        f"{choices}\n"
        "The best answer is:"
    )


def build_prompt(tokenizer, question: str, prompt_style: str = "native_short") -> str:
    """Render one video/user turn while leaving replacement to the processor."""
    if prompt_style == "native_short":
        text = f"{question.rstrip()}\n{ANSWER_INSTRUCTION}"
    elif prompt_style == "official_mvbench":
        text = official_mvbench_text(question)
    else:
        raise ValueError(f"unknown prompt style: {prompt_style}")
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": text},
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
    *,
    prompt_style: str = "native_short",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build model inputs equivalent to Qwen3VLProcessor for one preloaded video."""
    pixel_values, grid = native_preprocess_frames(
        frames, processor_config, min_pixels=min_pixels, max_pixels=max_pixels
    )
    prompt = build_prompt(tokenizer, question, prompt_style=prompt_style)
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
        "preprocess_min_pixels": int(
            processor_config["size"]["shortest_edge"] if min_pixels is None else min_pixels
        ),
        "preprocess_max_pixels": int(
            processor_config["size"]["longest_edge"] if max_pixels is None else max_pixels
        ),
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
    parser.add_argument(
        "--frame-policy", choices=["matched32", "official_2fps"], default="matched32"
    )
    parser.add_argument(
        "--prompt-style", choices=["native_short", "official_mvbench"],
        default="native_short",
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--timestamp-fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=OFFICIAL_MAX_FRAMES)
    parser.add_argument(
        "--max-total-video-tokens", type=int, default=OFFICIAL_MAX_TOTAL_VIDEO_TOKENS
    )
    parser.add_argument(
        "--max-tokens-per-unit", type=int, default=OFFICIAL_MAX_TOKENS_PER_UNIT
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="",
        help="optional HF device_map (e.g. auto) to shard model across GPUs; "
        "overrides --device placement for multi-GPU single-process eval",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    if args.num_frames < 2 or args.num_frames % 2:
        parser.error("num_frames must be an even integer >= 2")
    if args.timestamp_fps <= 0:
        parser.error("timestamp_fps must be positive")
    if args.max_frames < 4 or args.max_frames % 2:
        parser.error("max_frames must be an even integer >= 4")
    if args.max_total_video_tokens < 1 or args.max_tokens_per_unit < 1:
        parser.error("official visual-token budgets must be positive")
    if args.frame_policy == "official_2fps" and args.prompt_style != "official_mvbench":
        parser.error("official_2fps must use prompt_style=official_mvbench")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor_config = load_video_preprocess_config(args.model)
    load_kwargs = dict(
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    use_device_map = bool(args.device_map)
    if use_device_map:
        load_kwargs["device_map"] = args.device_map
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    overlay_audit = apply_native_overlay(model, args.overlay) if args.overlay else None
    model.requires_grad_(False).eval()
    if not use_device_map:
        model = model.to(args.device)
    # with device_map, inputs target the first parameter's device
    input_device = args.device if not use_device_map else str(next(model.parameters()).device)

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
            if args.frame_policy == "official_2fps":
                frames, frame_audit = load_official_frames(
                    item,
                    sample_fps=OFFICIAL_SAMPLE_FPS,
                    max_frames=args.max_frames,
                    seed=args.seed * 100003 + item_index,
                )
                max_pixels = official_max_pixels(
                    len(frames),
                    max_total_video_tokens=args.max_total_video_tokens,
                    max_tokens_per_unit=args.max_tokens_per_unit,
                )
            else:
                frames, frame_audit = load_native_frames(
                    item,
                    args.num_frames,
                    args.timestamp_fps,
                    args.seed * 100003 + item_index,
                )
                max_pixels = None
            inputs, native_audit = build_native_inputs(
                tokenizer,
                question,
                frames,
                float(frame_audit["timestamp_fps"]),
                processor_config,
                prompt_style=args.prompt_style,
                max_pixels=max_pixels,
            )
            grid = native_audit["video_grid_thw"]
            native_tokens = native_audit["native_video_tokens"]
            if args.frame_policy == "official_2fps":
                if native_tokens > args.max_total_video_tokens:
                    raise ValueError(
                        f"native video tokens {native_tokens} exceed total budget "
                        f"{args.max_total_video_tokens}"
                    )
                if native_tokens > int(grid[0]) * args.max_tokens_per_unit:
                    raise ValueError(
                        f"native video tokens {native_tokens} exceed per-unit budget "
                        f"{args.max_tokens_per_unit}"
                    )
            inputs = _move_inputs(inputs, input_device)
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
            "preprocess_max_pixels": native_audit["preprocess_max_pixels"],
            **frame_audit,
        })
        if (local_index + 1) % 50 == 0:
            correct = sum(record["ok"] for record in records)
            print(
                f"{local_index + 1}/{len(items)} acc="
                f"{100 * correct / max(len(records), 1):.2f}%"
            )

    if args.frame_policy == "official_2fps":
        frame_protocol = "official_budget_reproduction_2fps_real_video"
        answer_instruction = OFFICIAL_MVBENCH_INSTRUCTION
    else:
        frame_protocol = "matched_preextracted_32_native_compatible_smart_resize"
        answer_instruction = ANSWER_INSTRUCTION
    metadata = {
        "model": os.path.abspath(args.model),
        "overlay": overlay_audit,
        "frame_protocol": frame_protocol,
        "frame_policy": args.frame_policy,
        "prompt_style": args.prompt_style,
        "num_frames": args.num_frames,
        "max_frames": args.max_frames,
        "fallback_timestamp_fps": args.timestamp_fps,
        "requested_sample_fps": (
            OFFICIAL_SAMPLE_FPS if args.frame_policy == "official_2fps" else args.timestamp_fps
        ),
        "answer_instruction": answer_instruction,
        "official_budget": {
            "max_total_video_tokens": args.max_total_video_tokens,
            "max_tokens_per_unit": args.max_tokens_per_unit,
        } if args.frame_policy == "official_2fps" else None,
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
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False)
    print(
        f"{args.task}: {document['correct']}/{document['total']} "
        f"= {100 * document['acc']:.2f}%; skipped={skipped}; output={args.output}"
    )


if __name__ == "__main__":
    main()
