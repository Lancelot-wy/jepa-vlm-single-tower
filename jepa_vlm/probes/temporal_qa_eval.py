"""Held-out temporal-order QA evaluation for Phase B checkpoints (product-level readout).

Each val clip is shown once, either in its true order or corrupted (shuffle/reverse,
deterministic per index). The model is asked QAVideoDataset.TEMPORAL_Q and scored by
answer likelihood: predict "no" iff CE("no") < CE("yes"). Reports overall accuracy
plus per-corruption breakdown and N.

Compare arms trained in THIS token layout against each other (joint vs sft). The raw
base model was never trained with 4-token pooled frames, so its absolute number is
out-of-distribution and not meaningful.

  python -m jepa_vlm.probes.temporal_qa_eval --config <run>/config.json \
      --ckpt <run>/step_4000 --manifest data/llava_video/val.jsonl --max-clips 500
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data.datasets import QACollator, QAVideoDataset, load_manifest
from ..data.video_io import decode_frames, patchify, resize_center_crop
from .extract_features import load_run

ANSWERS = ("yes", "no")


def build_samples(items, cfg, seed: int):
    """Deterministic eval set: even index = true order ("yes"), odd = corrupted ("no",
    alternating shuffle / reverse)."""
    tc, mc = cfg.train, cfg.model
    for i, it in enumerate(items):
        rng = np.random.default_rng(seed * 100003 + i)
        frames = decode_frames(
            it["video"] if not tc.data_root else f"{tc.data_root}/{it['video']}",
            tc.num_frames, tc.sample_fps, tc.frame_sampling,
            start=it.get("start"), end=it.get("end"), random_offset=False, rng=rng,
        )
        corrupt = i % 2 == 1
        kind = "none"
        if corrupt:
            if (i // 2) % 2 == 0:
                frames, kind = frames[::-1].copy(), "reverse"
            else:
                perm = rng.permutation(len(frames))
                while (perm == np.arange(len(frames))).all():
                    perm = rng.permutation(len(frames))
                frames, kind = frames[perm].copy(), "shuffle"
        pixel_values, grid = patchify(resize_center_crop(frames, mc.frame_size), mc.duplicate_frames)
        yield {"pixel_values": pixel_values, "grid_thw": grid,
               "question": QAVideoDataset.TEMPORAL_Q, "answer": "",
               "truth": "no" if corrupt else "yes", "kind": kind}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="run config.json")
    ap.add_argument("--ckpt", default="", help="checkpoint dir; empty = untrained (OOD, see docstring)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-flow", type=float, default=0.0,
                    help="drop clips below this motion score (use the global threshold, "
                         "e.g. 8.42, so eval and training share one data standard)")
    ap.add_argument("--max-clips", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
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
                          cfg.train.num_frames * cfg.model.tokens_per_frame,
                          cfg.train.max_text_len)

    items = load_manifest(args.manifest, min_flow=args.min_flow)[: args.max_clips]
    stats = {"none": [0, 0], "shuffle": [0, 0], "reverse": [0, 0]}  # [correct, total]

    buf = []
    def flush(buf):
        if not buf:
            return
        ce = {}
        for ans in ANSWERS:
            batch = collator([{**s, "answer": ans} for s in buf])
            out = model(
                pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                grid_thw=batch["grid_thw"],
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                disable_mask=True,
            )
            ce[ans] = out.ce_per_sample.float().cpu()
        pred = ["no" if n < y else "yes" for y, n in zip(ce["yes"], ce["no"])]
        for s, p in zip(buf, pred):
            stats[s["kind"]][1] += 1
            stats[s["kind"]][0] += int(p == s["truth"])

    for s in build_samples(items, cfg, args.seed):
        buf.append(s)
        if len(buf) == args.batch_size:
            flush(buf)
            buf = []
            done = sum(v[1] for v in stats.values())
            if done % 100 < args.batch_size:
                print(f"{done}/{len(items)}")
    flush(buf)

    correct = sum(v[0] for v in stats.values())
    total = sum(v[1] for v in stats.values())
    import math
    se = math.sqrt(0.25 / total) * 100 if total else float("nan")
    print("\n=== temporal QA accuracy ===")
    print(f"overall: {correct}/{total} = {100 * correct / total:.2f}%  (binomial se ~{se:.1f}pp)")
    for kind in ("none", "shuffle", "reverse"):
        c, t = stats[kind]
        if t:
            print(f"  {kind:8s} {c}/{t} = {100 * c / t:.2f}%")


if __name__ == "__main__":
    main()
