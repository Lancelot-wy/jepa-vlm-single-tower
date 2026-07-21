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
import collections
import json
import os
import re

import numpy as np
import torch

from ..config import resolved_raw_num_frames, resolved_temporal_units, resolved_visual_tokens
from ..data.datasets import QACollator
from ..data.video_io import decode_frames, patchify, resize_center_crop
from .extract_features import load_run

# matches "(A) foo", "A. foo", "A) foo", "A: foo"
_OPT_RE = re.compile(r"^\s*\(?([A-H])[\).:.]\s*(.+?)\s*$")


def parse_options(question: str):
    """Return [(letter, full_line_as_answer), ...] parsed from the MCQ question text."""
    opts = []
    for line in question.splitlines():
        m = _OPT_RE.match(line)
        if m:
            opts.append((m.group(1).upper(), line.strip()))
    return opts


def target_letter(target: str):
    m = _OPT_RE.match(target.strip())
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*\(?([A-H])\b", target.strip())
    return m.group(1).upper() if m else None


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
    args = ap.parse_args()

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
    items = load_items(args.data, args.task, args.max_clips)
    print(f"{args.task}: {len(items)} items")

    stats = collections.defaultdict(lambda: [0, 0])  # subcat -> [correct, total]
    records = []  # per-sample, for paired significance tests across arms
    n_parsed = n_skip = 0
    for i, it in enumerate(items):
        question = it.get("问题", "")
        opts = parse_options(question)
        tgt = target_letter(str(it.get("目标值", "")))
        if len(opts) < 2 or tgt is None or tgt not in {o[0] for o in opts}:
            n_skip += 1
            continue
        n_parsed += 1
        meta = it.get("meta") or {}
        images_info = meta.get("images_info") or []
        rng = np.random.default_rng(args.seed * 100003 + i)
        try:
            if images_info:
                frames = load_image_frames(images_info, resolved_raw_num_frames(cfg))
            else:
                frames = decode_frames(
                    it["视频"], resolved_raw_num_frames(cfg), tc.sample_fps, tc.frame_sampling,
                    random_offset=False, rng=rng,
                )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] decode failed: {e}")
            n_skip += 1
            n_parsed -= 1
            continue
        pv, grid = patchify(
            resize_center_crop(frames, mc.frame_size), mc.duplicate_frames,
            tc.temporal_patch_size,
        )

        batch = collator([
            {"pixel_values": pv, "grid_thw": grid, "question": question, "answer": ans}
            for _, ans in opts
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
        stats[sub][1] += 1
        stats[sub][0] += int(pred == tgt)
        records.append({
            "idx": i, "pred": pred, "gold": tgt, "sub_type": sub,
            "ok": int(pred == tgt),
            "option_scores": {letter: float(score) for (letter, _), score in zip(opts, ce)},
        })
        if (i + 1) % 100 == 0:
            done = sum(v[1] for v in stats.values())
            corr = sum(v[0] for v in stats.values())
            print(f"{i + 1}/{len(items)}  running acc {100 * corr / max(done, 1):.2f}%")

    total = sum(v[1] for v in stats.values())
    correct = sum(v[0] for v in stats.values())
    print(f"\n=== {args.task} MCQ accuracy ===")
    print(f"parsed {n_parsed}, skipped {n_skip}")
    if total:
        print(f"overall: {correct}/{total} = {100 * correct / total:.2f}%")
    for sub in sorted(stats):
        c, t = stats[sub]
        print(f"  {sub:28s} {c}/{t} = {100 * c / t:.2f}%")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"task": args.task, "acc": correct / max(total, 1),
                       "correct": correct, "total": total,
                       "skipped": n_skip, "results": records}, f, ensure_ascii=False)
        print(f"per-sample records -> {args.output}")


if __name__ == "__main__":
    main()
