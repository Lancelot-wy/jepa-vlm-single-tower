"""Non-triviality check (plan eval #3, the go/no-go gate).

Compares the model's masked-position regression MSE against the trivial baseline of
copying the nearest unmasked frame's (normed) features. If the model is not clearly
better, it has learned the "copy the neighbour" shortcut and the run FAILS.

  python -m jepa_vlm.probes.nontrivial_check --config runs/x/config.json \
      --ckpt runs/x/step_20000 --manifest data/ssv2/val.jsonl [--batches 100]
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..data.datasets import ManifestVideoDataset, collate_visual
from .extract_features import load_run

PASS_MARGIN = 0.8  # reg_loss / copy_mse must be below this to count as non-trivial


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    cfg, model = load_run(args.config, args.ckpt or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dtype = next(model.parameters()).dtype

    ds = ManifestVideoDataset(
        args.manifest, data_root=args.data_root or cfg.train.data_root,
        num_frames=cfg.train.num_frames, sample_fps=cfg.train.sample_fps,
        frame_sampling=cfg.train.frame_sampling, frame_size=cfg.model.frame_size,
        duplicate_frames=cfg.model.duplicate_frames, training=False,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=collate_visual, drop_last=True)

    gen = torch.Generator().manual_seed(0)  # fixed masks -> comparable across checkpoints
    reg, copy, adj, std = [], [], [], []
    for bi, batch in enumerate(dl):
        if bi >= args.batches:
            break
        out = model(batch["pixel_values"].to(device, dtype=dtype), batch["grid_thw"], generator=gen)
        reg.append(out.metrics["reg_loss"])
        if "copy_mse" in out.metrics:
            copy.append(out.metrics["copy_mse"])
        adj.append(out.metrics["adj_cos"])
        std.append(out.metrics["target_std"])

    reg_m = sum(reg) / len(reg)
    copy_m = sum(copy) / len(copy) if copy else float("nan")
    ratio = reg_m / copy_m if copy else float("nan")
    print("\n=== non-triviality check ===")
    print(f"masked reg MSE      : {reg_m:.5f}")
    print(f"copy-baseline MSE   : {copy_m:.5f}")
    print(f"ratio (reg/copy)    : {ratio:.3f}   (pass if < {PASS_MARGIN})")
    print(f"target_std          : {sum(std)/len(std):.4f}  (collapse if -> 0)")
    print(f"adjacent-frame cos  : {sum(adj)/len(adj):.4f}  (trivial targets if -> 1)")
    if ratio < PASS_MARGIN:
        print("VERDICT: PASS - model beats the copy baseline")
    else:
        print("VERDICT: FAIL - trivial solution; raise mask_ratio / lower sample_fps (plan section 6)")


if __name__ == "__main__":
    main()
