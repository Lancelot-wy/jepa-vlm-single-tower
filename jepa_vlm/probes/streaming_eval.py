"""Streaming-protocol MCQ eval on OVO-Bench / StreamingBench for JEPA-VLM ckpts.

Protocol: each question carries a stream timestamp t (OVO `realtime`, StreamingBench
`time_stamp`). The model may only see video content from [0, t] — we decode a frame
context from the prefix and score options by answer likelihood (same likelihood
scoring as mcq_eval.py: lowest mean per-token CE wins).

Context modes:
  recent : num_frames sampled uniformly from the last --window seconds before t
           (SimpleStream-style recent window — the baseline every arm must beat)
  prefix : num_frames sampled uniformly from the whole [0, t] prefix
           (bounded "full history" reference under the same frame budget)

Benchmarks:
  ovo : --data <ovo_bench_new.json> --video-root <dir with chunked_videos/<id>.mp4>
        --tasks EPM,ASI (backward tracing; HLD excluded by default: measures refusal,
        not memory — see gate experiment 2026-07-09)
  sb  : --data <task_csv[,csv2...]> --video-root <dir>
        question_id "<Task>_sample_<N>_<i>" -> video "<video-root>/sample_<N>/video.mp4"
        (point --video-root at the extracted folder of the matching task zip)

Usage (single GPU):
  python -m jepa_vlm.probes.streaming_eval --config <run>/config.json --ckpt <run>/step_4000 \
      --bench ovo --data $OVO/ovo_bench_new.json --video-root $OVO --tasks EPM,ASI \
      --mode recent --window 64 --out results/streaming/ovo_dv25_recent.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import os
import re

import numpy as np
import torch

from ..config import resolved_raw_num_frames, resolved_temporal_units, resolved_visual_tokens

from ..data.datasets import QACollator
from ..data.video_io import patchify, resize_center_crop
from .extract_features import load_run

LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------- decode
def decode_prefix_frames(path: str, t_end: float, num_frames: int,
                         mode: str, window: float) -> np.ndarray:
    """Decode `num_frames` uniformly from [t0, t_end] of the video, where
    t0 = max(0, t_end-window) for mode=recent and 0 for mode=prefix.
    Never reads past t_end (streaming causality)."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration \
            else (float(container.duration) / av.time_base if container.duration else None)
        if duration is not None:
            t_end = min(t_end, duration)
        t_end = max(t_end, 0.5)
        t0 = 0.0 if mode == "prefix" else max(0.0, t_end - window)
        want = np.linspace(t0, max(t0, t_end - 1e-3), num_frames)

        # single sequential decode pass (chunked prefixes are short; avoids seek bugs)
        frames, wi = [], 0
        for frame in container.decode(stream):
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if ts is None:
                continue
            if ts > t_end + 0.5:
                break
            while wi < len(want) and ts >= want[wi] - 1e-6:
                frames.append(frame.to_ndarray(format="rgb24"))
                wi += 1
            if wi >= len(want):
                break
        if not frames:
            raise RuntimeError(f"no frames decoded in [0,{t_end:.1f}]s of {path}")
        while len(frames) < num_frames:          # pad by repeating the last frame
            frames.append(frames[-1])
    return np.stack(frames[:num_frames])


# --------------------------------------------------------------------- loaders
def _norm_gt(gt, options) -> str | None:
    if isinstance(gt, int):
        return LETTERS[gt] if 0 <= gt < len(options) else None
    s = str(gt).strip()
    m = re.match(r"\(?([A-H])\b", s)
    if m:
        return m.group(1).upper()
    for i, o in enumerate(options):              # match by option text
        if s and s.lower() == str(o).strip().lower():
            return LETTERS[i]
    return None


def _mcq_texts(question: str, options: list[str]):
    """Build the MCQ prompt and per-option answer strings ('A. <text>')."""
    lines, answers = [], []
    for i, opt in enumerate(options):
        opt = str(opt).strip()
        if re.match(r"^\(?[A-H][\).:.]", opt):   # already lettered
            line = opt
        else:
            line = f"{LETTERS[i]}. {opt}"
        lines.append(line)
        answers.append(line)
    q = question.strip() + "\n" + "\n".join(lines) + \
        "\nAnswer with the option's letter from the given choices directly."
    return q, answers


def load_ovo(path: str, video_root: str, tasks: set[str], max_items: int):
    items = []
    for d in json.load(open(path)):
        if tasks and d.get("task") not in tasks:
            continue
        if "realtime" not in d or "options" not in d:
            continue
        gt = _norm_gt(d.get("gt"), d["options"])
        if gt is None:
            continue
        video = os.path.join(video_root, "chunked_videos", f"{d['id']}.mp4")
        items.append(dict(qid=f"ovo_{d['id']}", task=d["task"], video=video,
                          t=float(d["realtime"]), question=d["question"],
                          options=[str(o) for o in d["options"]], gt=gt))
        if max_items and len(items) >= max_items:
            break
    return items


def _hms_to_sec(ts: str) -> float:
    parts = [float(x) for x in str(ts).strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_sb(csv_paths: str, video_root: str, tasks: set[str], max_items: int):
    items = []
    for cp in csv_paths.split(","):
        with open(cp.strip(), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if tasks and row.get("task_type") and row["task_type"] not in tasks:
                    continue
                qid = row["question_id"]
                m = re.search(r"sample_(\d+)", qid)
                if not m:
                    continue
                try:
                    options = ast.literal_eval(row["options"])
                except (ValueError, SyntaxError):
                    continue
                gt = _norm_gt(row.get("answer", ""), options)
                if gt is None:
                    continue
                video = os.path.join(video_root, f"sample_{m.group(1)}", "video.mp4")
                items.append(dict(qid=f"sb_{qid}", task=row.get("task_type", "?"),
                                  video=video, t=_hms_to_sec(row["time_stamp"]),
                                  question=row["question"],
                                  options=[str(o) for o in options], gt=gt))
                if max_items and len(items) >= max_items:
                    return items
    return items


# --------------------------------------------------------------------- main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="", help="checkpoint dir; empty = base (untrained wrapper)")
    ap.add_argument("--bench", required=True, choices=["ovo", "sb"])
    ap.add_argument("--data", required=True, help="ovo json / comma-separated sb csv paths")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--tasks", default="", help="comma filter, e.g. EPM,ASI (ovo) — empty = all")
    ap.add_argument("--mode", default="recent", choices=["recent", "prefix"])
    ap.add_argument("--window", type=float, default=64.0, help="recent-window seconds")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--min-t", type=float, default=30.0, help="skip questions earlier than this")
    ap.add_argument("--out", required=True, help="per-question jsonl (append-safe resume)")
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    if args.bench == "ovo":
        items = load_ovo(args.data, args.video_root, tasks, args.max_items)
    else:
        items = load_sb(args.data, args.video_root, tasks, args.max_items)
    items = [it for it in items if it["t"] >= args.min_t]
    print(f"{args.bench}: {len(items)} questions (mode={args.mode}, window={args.window})")

    done = set()
    if os.path.exists(args.out):                 # resume: skip already-answered qids
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} already done")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg, model = load_run(args.config, args.ckpt or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dtype = next(model.parameters()).dtype
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained)
    ids = {k: getattr(model.hf_config, k) for k in
           ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
    collator = QACollator(tokenizer, ids,
                          resolved_temporal_units(cfg) * resolved_visual_tokens(cfg),
                          cfg.train.max_text_len)
    tc, mc = cfg.train, cfg.model

    stats = collections.defaultdict(lambda: [0, 0])
    fout = open(args.out, "a")
    n_fail = 0
    for i, it in enumerate(items):
        if it["qid"] in done:
            continue
        try:
            frames = decode_prefix_frames(
                it["video"], it["t"], resolved_raw_num_frames(cfg),
                args.mode, args.window,
            )
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 10:
                print(f"  [{i}] decode failed ({it['video']}): {e}")
            continue
        q, answers = _mcq_texts(it["question"], it["options"])
        pv, grid = patchify(
            resize_center_crop(frames, mc.frame_size), mc.duplicate_frames,
            tc.temporal_patch_size,
        )
        batch = collator([
            {"pixel_values": pv, "grid_thw": grid, "question": q, "answer": ans}
            for ans in answers
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
        pred = LETTERS[int(torch.argmin(ce))]
        rec = dict(qid=it["qid"], task=it["task"], t=it["t"], mode=args.mode,
                   window=args.window, pred=pred, gt=it["gt"],
                   correct=int(pred == it["gt"]),
                   ce=[round(float(c), 4) for c in ce])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        stats[it["task"]][1] += 1
        stats[it["task"]][0] += rec["correct"]
        if (i + 1) % 50 == 0:
            tot = sum(v[1] for v in stats.values())
            cor = sum(v[0] for v in stats.values())
            print(f"{i + 1}/{len(items)}  running acc {100 * cor / max(tot, 1):.2f}%")
    fout.close()

    total = sum(v[1] for v in stats.values())
    correct = sum(v[0] for v in stats.values())
    print(f"\n=== {args.bench} streaming MCQ ({args.mode}) ===")
    print(f"decode failures: {n_fail}")
    if total:
        print(f"overall (this run): {correct}/{total} = {100 * correct / total:.2f}%")
    for t in sorted(stats):
        c, n = stats[t]
        print(f"  {t:12s} {c}/{n} = {100 * c / n:.2f}%")


if __name__ == "__main__":
    main()
