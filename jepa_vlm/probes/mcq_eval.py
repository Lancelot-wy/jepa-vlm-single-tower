"""Multiple-choice video-benchmark eval (MVBench / TempCompass) for JEPA-VLM ckpts.

The single-tower model is not vLLM-servable, so we score choices by answer
likelihood instead of free generation: for each option we run a no-mask forward
with that option as the answer and take the mean per-token CE (JepaOutput.ce_per_sample,
already length-normalized). The option with the lowest CE is the prediction; it is
scored "correct" iff its letter matches the target letter parsed from 目标值.

Reads the merged offline-eval jsonl (Chinese fields 任务类别/问题/目标值/视频/meta) and
filters to one --task. Reports overall + per-子类别 accuracy.

  python -m jepa_vlm.probes.mcq_eval --config <run>/config.json --ckpt <run>/step_4000 \
      --data <merged>.jsonl --task MVBench --max-clips 500
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import torch

from ..config import resolved_raw_num_frames, resolved_temporal_units, resolved_visual_tokens
from ..data.datasets import QACollator
from ..data.video_io import decode_frames, patchify, resize_center_crop
from .extract_features import load_run
from .mcq_utils import parse_options, result_document, target_letter


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_image_frames(images_info: list, num_frames: int) -> np.ndarray:
    """Load the benchmark's pre-extracted jpg frames as (T,H,W,3) uint8, uniformly
    sampling `num_frames` of them. These are the frames the offline-eval framework
    feeds the model; using them avoids re-decoding the (already-trimmed) mp4 whose
    Charades start/end bounds refer to the *original* untrimmed video and overrun EOF."""
    from PIL import Image

    paths = [f["image"] for f in images_info if f.get("image")]
    if not paths:
        raise RuntimeError("empty images_info")
    idx = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return np.stack([np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx])


def load_items(path: str, task: str, max_clips: int):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("任务类别") != task:
                continue
            items.append(d)
            if max_clips and len(items) >= max_clips:
                break
    return items


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="run config.json")
    ap.add_argument("--ckpt", default="", help="checkpoint dir; empty = untrained (OOD)")
    ap.add_argument("--data", required=True, help="merged offline-eval jsonl")
    ap.add_argument("--task", required=True, help="任务类别, e.g. MVBench or Tempcompass")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="", help="write per-sample records json (paired tests)")
    ap.add_argument(
        "--answer-format", choices=["full_option", "letter"], default="full_option",
        help="candidate answer text; full_option preserves the historical evaluator",
    )
    ap.add_argument("--protocol", default="custom_answer_likelihood")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        ap.error("require num_shards >= 1 and 0 <= shard_index < num_shards")

    cfg, model = load_run(args.config, args.ckpt or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dtype = next(model.parameters()).dtype

    if cfg.model.tiny_config:
        from ..train import ByteTokenizer
        tokenizer = ByteTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained)
    ids = {k: getattr(model.hf_config, k) for k in
           ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
    collator = QACollator(tokenizer, ids,
                          resolved_temporal_units(cfg) * resolved_visual_tokens(cfg),
                          cfg.train.max_text_len)

    tc, mc = cfg.train, cfg.model
    all_items = load_items(args.data, args.task, args.max_clips)
    items = [
        (index, item) for index, item in enumerate(all_items)
        if index % args.num_shards == args.shard_index
    ]
    print(
        f"{args.task}: {len(items)}/{len(all_items)} items "
        f"(shard {args.shard_index}/{args.num_shards}, answer={args.answer_format})"
    )

    records = []  # per-sample, for paired significance tests across arms
    n_parsed = n_skip = 0
    for local_index, (item_index, it) in enumerate(items):
        question = it.get("问题", "")
        opts = parse_options(question)
        tgt = target_letter(str(it.get("目标值", "")))
        if len(opts) < 2 or tgt is None or tgt not in {o[0] for o in opts}:
            n_skip += 1
            continue
        n_parsed += 1
        meta = it.get("meta") or {}
        images_info = meta.get("images_info") or []
        rng = np.random.default_rng(args.seed * 100003 + item_index)
        try:
            if images_info:
                frames = load_image_frames(images_info, resolved_raw_num_frames(cfg))
            else:
                frames = decode_frames(
                    it["视频"], resolved_raw_num_frames(cfg), tc.sample_fps, tc.frame_sampling,
                    random_offset=False, rng=rng,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  [{item_index}] decode failed: {e}")
            n_skip += 1
            n_parsed -= 1
            continue
        pv, grid = patchify(
            resize_center_crop(frames, mc.frame_size), mc.duplicate_frames,
            tc.temporal_patch_size,
        )

        candidate_answers = [
            full_line if args.answer_format == "full_option" else letter
            for letter, full_line in opts
        ]
        batch = collator([
            {"pixel_values": pv, "grid_thw": grid, "question": question, "answer": answer}
            for answer in candidate_answers
        ])
        out = model(
            pixel_values=batch["pixel_values"].to(device, dtype=dtype),
            grid_thw=batch["grid_thw"],
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
            disable_mask=True,
        )
        ce = out.ce_per_sample.float().cpu()
        pred = opts[int(torch.argmin(ce))][0]
        sub = it.get("子类别", "?")
        records.append({
            "idx": item_index, "pred": pred, "gold": tgt, "sub_type": sub,
            "ok": int(pred == tgt),
            "option_scores": {letter: float(score) for (letter, _), score in zip(opts, ce)},
        })
        if (local_index + 1) % 100 == 0:
            corr = sum(record["ok"] for record in records)
            print(
                f"{local_index + 1}/{len(items)}  "
                f"running acc {100 * corr / max(len(records), 1):.2f}%"
            )

    total = len(records)
    correct = sum(record["ok"] for record in records)
    document = result_document(
        task=args.task,
        protocol=args.protocol,
        scoring=f"answer_likelihood_mean_token_ce:{args.answer_format}",
        records=records,
        skipped=n_skip,
        metadata={
            "answer_format": args.answer_format,
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.ckpt) if args.ckpt else None,
            "dataset": os.path.abspath(args.data),
            "visual_tokens_per_unit": resolved_visual_tokens(cfg),
            "temporal_units": resolved_temporal_units(cfg),
            "raw_num_frames": resolved_raw_num_frames(cfg),
            "evaluator_commit": _git_commit(),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "max_clips": args.max_clips,
        },
    )
    print(f"\n=== {args.task} MCQ accuracy ===")
    print(f"parsed {n_parsed}, skipped {n_skip}")
    if total:
        print(f"overall: {correct}/{total} = {100 * correct / total:.2f}%")
    for sub, values in document["categories"].items():
        print(
            f"  {sub:28s} {values['correct']}/{values['total']} "
            f"= {100 * values['acc']:.2f}%"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(document, f, ensure_ascii=False)
        print(f"per-sample records -> {args.output}")


if __name__ == "__main__":
    main()
