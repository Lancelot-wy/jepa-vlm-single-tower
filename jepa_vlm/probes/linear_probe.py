"""Linear probe on extracted features (plan eval #1: SSv2 classes; #2: shuffle/reverse).

  python -m jepa_vlm.probes.linear_probe --train feats/train.pt --val feats/val.pt \
      --feature layer18_clip
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def train_probe(x_tr, y_tr, x_va, y_va, epochs=30, lr=1e-3, wd=1e-4, device="cpu"):
    x_tr, y_tr, x_va, y_va = x_tr.to(device), y_tr.to(device), x_va.to(device), y_va.to(device)
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_va = (x_tr - mu) / sd, (x_va - mu) / sd
    n_cls = int(max(y_tr.max(), y_va.max())) + 1
    w = torch.nn.Linear(x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    n = len(x_tr)
    bs = min(1024, n)
    best = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        w.train()
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            loss = F.cross_entropy(w(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        w.eval()
        with torch.no_grad():
            acc = (w(x_va).argmax(-1) == y_va).float().mean().item()
        best = max(best, acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--feature", default="", help="e.g. layer18_clip; default: all available")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    tr = torch.load(args.train, weights_only=False)
    va = torch.load(args.val, weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = [args.feature] if args.feature else [k for k in tr if k.startswith("layer") and "_" in k]
    results = {}
    for k in keys:
        acc = train_probe(tr[k], tr["labels"], va[k], va["labels"],
                          epochs=args.epochs, lr=args.lr, device=device)
        results[k] = acc
        print(f"{k}: top-1 = {acc:.4f}")
    print("\n=== linear probe summary ===")
    for k, v in results.items():
        print(f"{k:24s} {v * 100:.2f}%")


if __name__ == "__main__":
    main()
