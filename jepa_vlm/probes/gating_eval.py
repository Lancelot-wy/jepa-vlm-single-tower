"""L1 gating eval: can a cheap latent-surprise signal decide WHEN to wake the VLM?

The VLM itself is never trained or modified. A tiny GRU predictor (<10M params,
trained in minutes on pooled per-chunk latents of held-out videos) produces a
surprise signal; the gate triggers a "VLM wake-up" on high-surprise chunks. When a
question arrives at stream time t, the model answers from a recent window anchored
at the LATEST TRIGGER <= t (fewer wake-ups => staler context). We sweep the trigger
budget and compare accuracy-vs-wakeups curves across gate signals:

  surprise : GRU next-latent prediction error (ours; the only learned component)
  framediff: mean abs pixel diff between chunk boundaries (naive gate to beat)
  periodic : every k-th chunk
  always   : anchor == t (upper bound; equals plain streaming recent-window eval)

Judgment (pre-registered): surprise must beat framediff/periodic at equal budget;
otherwise the gating story is dead (L0/L1 negative results already cover selection).

  python -m jepa_vlm.probes.gating_eval --config <run>/config.json \
      --bench sb --data $SB/Real_Time_Visual_Understanding.csv --video-root $SB/real \
      --train-videos 20 --max-items 200 --budgets 0.1,0.25,0.5 \
      --out results/gating/sb_rtvu.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch
import torch.nn as nn

from ..data.video_io import patchify, resize_center_crop
from .extract_features import load_run
from .streaming_eval import (LETTERS, _mcq_texts, decode_prefix_frames,
                             load_ovo, load_sb)


# ------------------------------------------------------------------ chunk latents
@torch.no_grad()
def video_chunk_latents(model, path: str, t_end: float, chunk_sec: float,
                        frames_per_chunk: int, frame_size: int, duplicate: bool,
                        device, dtype, cache_dir: str | None = None):
    """Encode the [0, t_end] prefix as consecutive chunks; return
    (latents (N, D) pooled-mean visual latents, framediff (N,) boundary pixel diff)."""
    key = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = os.path.join(cache_dir, os.path.basename(os.path.dirname(path)) + "_" +
                           f"{int(t_end)}_{chunk_sec}.npz")
        if os.path.exists(key):
            z = np.load(key)
            return z["lat"], z["fd"]
    lats, fds, prev_last = [], [], None
    n_chunks = max(1, int(t_end // chunk_sec))
    for ci in range(n_chunks):
        t1 = min((ci + 1) * chunk_sec, t_end)
        frames = decode_prefix_frames(path, t1, frames_per_chunk, "recent", chunk_sec)
        pv, grid = patchify(resize_center_crop(frames, frame_size), duplicate)
        h, _ = model.encode_video(pv[None].to(device, dtype=dtype), grid)
        lats.append(h.float().mean(dim=(1, 2)).squeeze(0).cpu().numpy())   # (D,)
        cur_first, cur_last = frames[0].astype(np.float32), frames[-1].astype(np.float32)
        fds.append(0.0 if prev_last is None else float(np.abs(cur_first - prev_last).mean()))
        prev_last = cur_last
    lat = np.stack(lats)
    fd = np.asarray(fds)
    if key:
        np.savez(key, lat=lat, fd=fd)
    return lat, fd


# ------------------------------------------------------------------ tiny predictor
class TinyGRU(nn.Module):
    def __init__(self, d_in: int, d_h: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.gru = nn.GRU(d_h, d_h, batch_first=True)
        self.out = nn.Linear(d_h, d_in)

    def forward(self, x):                        # x (B, N, D) -> pred for steps 1..N
        z, _ = self.gru(self.proj(x))
        return self.out(z)


def train_predictor(seqs: list[np.ndarray], epochs: int = 30, device="cuda"):
    d = seqs[0].shape[-1]
    net = TinyGRU(d).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    xs = [torch.tensor((s - s.mean(0)) / (s.std(0) + 1e-6), dtype=torch.float32)
          for s in seqs if len(s) >= 4]
    for ep in range(epochs):
        tot = n = 0
        for x in xs:
            x = x[None].to(device)
            pred = net(x[:, :-1])
            loss = nn.functional.mse_loss(pred, x[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        if (ep + 1) % 10 == 0:
            # copy-last reference on the same sequences
            cl = float(np.mean([np.mean((x[0, 1:].cpu().numpy() - x[0, :-1].cpu().numpy()) ** 2)
                                for x in [t[None] for t in [xx[0].cpu() for xx in [x]]]]))
            print(f"  predictor ep{ep + 1}: mse {tot / max(n, 1):.4f}")
    return net


def surprise_signal(net, lat: np.ndarray, device="cuda") -> np.ndarray:
    x = torch.tensor((lat - lat.mean(0)) / (lat.std(0) + 1e-6), dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        pred = net(x[:, :-1])
        err = ((pred - x[:, 1:]) ** 2).mean(-1).squeeze(0).cpu().numpy()
    return np.concatenate([[err.max() if len(err) else 1.0], err])   # chunk0 = always novel


# ------------------------------------------------------------------ gate simulation
def anchors_from_triggers(trig_times: np.ndarray, t: float, chunk_sec: float) -> float:
    """Latest trigger time <= t (fall back to first chunk end)."""
    ok = trig_times[trig_times <= t + 1e-6]
    return float(ok.max()) if len(ok) else chunk_sec


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--bench", required=True, choices=["ovo", "sb"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--chunk-sec", type=float, default=4.0)
    ap.add_argument("--frames-per-chunk", type=int, default=4)
    ap.add_argument("--train-videos", type=int, default=20,
                    help="first N distinct videos train the tiny predictor (excluded from eval)")
    ap.add_argument("--budgets", default="0.1,0.25,0.5", help="fraction of chunks that may trigger")
    ap.add_argument("--window", type=float, default=64.0)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--min-t", type=float, default=30.0)
    ap.add_argument("--cache", default="results/gating/latent_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    loader = load_ovo if args.bench == "ovo" else load_sb
    items = [it for it in loader(args.data, args.video_root, tasks, 0)
             if it["t"] >= args.min_t]

    by_video = collections.defaultdict(list)
    for it in items:
        by_video[it["video"]].append(it)
    videos = sorted(by_video)
    train_vids, eval_vids = videos[:args.train_videos], videos[args.train_videos:]
    eval_items = [it for v in eval_vids for it in by_video[v]][: args.max_items]
    print(f"videos: {len(train_vids)} predictor-train / {len(eval_vids)} eval; "
          f"{len(eval_items)} questions")

    cfg, model = load_run(args.config, args.ckpt or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dtype = next(model.parameters()).dtype
    tc, mc = cfg.train, cfg.model

    def latents_for(video, t_end):
        return video_chunk_latents(model, video, t_end, args.chunk_sec,
                                   args.frames_per_chunk, mc.frame_size,
                                   mc.duplicate_frames, device, dtype, args.cache)

    # 1) predictor on train videos
    seqs = []
    for v in train_vids:
        t_end = max(it["t"] for it in by_video[v])
        lat, _ = latents_for(v, t_end)
        seqs.append(lat)
    net = train_predictor(seqs, device=device)

    # 2) signals per eval video
    sig = {}
    for v in eval_vids:
        t_end = max(it["t"] for it in by_video[v])
        lat, fd = latents_for(v, t_end)
        sig[v] = dict(surprise=surprise_signal(net, lat, device), framediff=fd,
                      n=len(lat))

    # 3) QA scoring with per-(signal,budget) anchors
    from transformers import AutoTokenizer
    from ..data.datasets import QACollator
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained)
    ids = {k: getattr(model.hf_config, k) for k in
           ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
    collator = QACollator(tokenizer, ids,
                          tc.num_frames * mc.tokens_per_frame, tc.max_text_len)

    def score(video, t_anchor, question, options):
        frames = decode_prefix_frames(video, t_anchor, tc.num_frames, "recent", args.window)
        q, answers = _mcq_texts(question, options)
        pv, grid = patchify(resize_center_crop(frames, mc.frame_size), mc.duplicate_frames)
        batch = collator([{"pixel_values": pv, "grid_thw": grid, "question": q, "answer": a}
                          for a in answers])
        out = model(pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                    grid_thw=batch["grid_thw"], input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device), disable_mask=True)
        return LETTERS[int(torch.argmin(out.ce_per_sample.float().cpu()))]

    budgets = [float(b) for b in args.budgets.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a")
    table = collections.defaultdict(lambda: [0, 0, 0.0])   # (signal,budget)->[corr,tot,trigs]

    for i, it in enumerate(eval_items):
        v = it["video"]
        s = sig[v]
        n = s["n"]
        chunk_ends = (np.arange(n) + 1) * args.chunk_sec
        conds = {}
        for b in budgets:
            k = max(1, int(round(b * n)))
            for name in ("surprise", "framediff"):
                idx = np.argsort(-s[name])[:k]
                conds[(name, b)] = chunk_ends[np.sort(idx)]
            step = max(1, int(round(1 / b)))
            conds[("periodic", b)] = chunk_ends[::step]
        conds[("always", 1.0)] = chunk_ends
        for (name, b), trig in conds.items():
            t_anchor = anchors_from_triggers(np.asarray(trig), it["t"], args.chunk_sec)
            try:
                pred = score(v, t_anchor, it["question"], it["options"])
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] score failed: {e}")
                continue
            correct = int(pred == it["gt"])
            table[(name, b)][0] += correct
            table[(name, b)][1] += 1
            table[(name, b)][2] += len(np.asarray(trig)[np.asarray(trig) <= it["t"] + 1e-6])
            fout.write(json.dumps(dict(qid=it["qid"], signal=name, budget=b,
                                       t=it["t"], anchor=t_anchor, pred=pred,
                                       gt=it["gt"], correct=correct),
                                  ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(eval_items)} questions done")
    fout.close()

    print("\n=== gating: accuracy vs wake-up budget ===")
    print(f"{'signal':10s} {'budget':>6s} {'acc':>8s} {'avg wakeups/q':>14s}")
    for (name, b) in sorted(table):
        c, t, tr = table[(name, b)]
        if t:
            print(f"{name:10s} {b:6.2f} {100 * c / t:7.2f}% {tr / t:14.1f}")


if __name__ == "__main__":
    main()
