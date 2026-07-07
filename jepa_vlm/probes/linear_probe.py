"""Linear probe on extracted features (plan eval #1: class probe; #2: shuffle/reverse).

Reports val-set size and, with --repeats > 1, mean +/- std over probe-training seeds.
Cross-arm deltas of 1-4pp only mean something relative to that spread (Round-2 base
probe reruns varied ~0.6pp), so use --repeats 5 for any go/no-go readout.

  python -m jepa_vlm.probes.linear_probe --train feats/train.pt --val feats/val.pt \
      --feature layer27_frames --repeats 5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def train_probe(x_tr, y_tr, x_va, y_va, epochs=30, lr=1e-3, wd=1e-4, device="cpu", seed=0):
    torch.manual_seed(seed)
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
    ap.add_argument("--repeats", type=int, default=1,
                    help="probe-training repeats with different seeds; report mean +/- std")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = torch.load(args.train, weights_only=False)
    va = torch.load(args.val, weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = [args.feature] if args.feature else [k for k in tr if k.startswith("layer") and "_" in k]

    n_tr, n_va = len(tr["labels"]), len(va["labels"])
    n_cls = int(max(tr["labels"].max(), va["labels"].max())) + 1
    # binomial se on the val set: the floor of resolvable differences between arms
    import math
    se = math.sqrt(0.25 / n_va) * 100
    print(f"train {n_tr} / val {n_va} clips, {n_cls} classes; "
          f"val binomial se <= {se:.2f}pp -> treat deltas < ~{2 * se:.1f}pp as noise")

    results = {}
    for k in keys:
        accs = [
            train_probe(tr[k], tr["labels"], va[k], va["labels"],
                        epochs=args.epochs, lr=args.lr, device=device, seed=args.seed + i)
            for i in range(max(args.repeats, 1))
        ]
        t = torch.tensor(accs) * 100
        results[k] = (float(t.mean()), float(t.std()) if len(accs) > 1 else 0.0)
        runs = " ".join(f"{a * 100:.2f}" for a in accs)
        print(f"{k}: top-1 = {t.mean():.2f}%" + (f" +/- {t.std():.2f} ({runs})" if len(accs) > 1 else ""))

    print("\n=== linear probe summary ===")
    for k, (m, s) in results.items():
        print(f"{k:24s} {m:.2f}%" + (f" +/- {s:.2f}" if args.repeats > 1 else ""))


if __name__ == "__main__":
    main()
