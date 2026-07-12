"""Extract LLM hidden-state features for linear probes (plan eval #1 and #2).

Runs a no-mask forward and saves, per configured layer (default: middle + last),
  - "clip":  mean over all visual tokens        -> (N, D)      for class probes
  - "frames": mean per latent frame, concat     -> (N, T*D)    for temporal-order probes
together with labels.

  python -m jepa_vlm.probes.extract_features --config runs/x/config.json \
      --ckpt runs/x/step_20000 --manifest data/ssv2/val.jsonl --out feats/ssv2_val.pt \
      [--temporal-transform random_shuffle|random_reverse] [--layers mid,last]
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from ..config import Config, ModelConfig, TrainConfig
from ..data.datasets import ManifestVideoDataset, collate_visual
from ..modeling.model import build_model


def load_run(config_path: str, ckpt_dir: str | None):
    if config_path.endswith((".yaml", ".yml")):
        from ..config import load_config
        cfg = load_config(config_path)
    else:
        with open(config_path) as f:
            d = json.load(f)
        cfg = Config(model=ModelConfig(**d["model"]), train=TrainConfig(**d["train"]))
    model = build_model(cfg)
    if ckpt_dir:
        state = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        print(f"loaded {len(state['model'])} tensors from {ckpt_dir}")
    model.eval()
    return cfg, model


def resolve_layers(spec: str, num_layers: int) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "mid":
            out.append(num_layers // 2)
        elif tok == "last":
            out.append(num_layers - 1)
        else:
            out.append(int(tok) % num_layers)
    return sorted(set(out))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.json of the run (or a yaml)")
    ap.add_argument("--ckpt", default="", help="checkpoint dir with state.pt; empty = base model (baseline)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="mid,last")
    ap.add_argument("--temporal-transform", default="none",
                    choices=["none", "random_shuffle", "random_reverse"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    if args.config.endswith((".yaml", ".yml")):
        from ..config import load_config
        cfg = load_config(args.config)
        model = build_model(cfg)
        model.eval()
    else:
        cfg, model = load_run(args.config, args.ckpt or None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    dtype = next(model.parameters()).dtype

    ds = ManifestVideoDataset(
        args.manifest, data_root=args.data_root or cfg.train.data_root,
        num_frames=cfg.train.num_frames, sample_fps=cfg.train.sample_fps,
        frame_sampling=cfg.train.frame_sampling, frame_size=cfg.model.frame_size,
        duplicate_frames=cfg.model.duplicate_frames, training=False,
        temporal_transform=args.temporal_transform,
    )
    if args.max_clips:
        ds.items = ds.items[: args.max_clips]
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=collate_visual)

    num_layers = model.hf_config.text_config.num_hidden_layers
    layers = resolve_layers(args.layers, num_layers)
    feats = {li: {"clip": [], "frames": []} for li in layers}
    labels = []
    for bi, batch in enumerate(dl):
        f = model.extract_features(
            batch["pixel_values"].to(device, dtype=dtype), batch["grid_thw"], layers)
        for li, x in f.items():  # (B, T, P, D)
            x = x.float()
            feats[li]["clip"].append(x.mean(dim=(1, 2)).cpu())
            feats[li]["frames"].append(x.mean(dim=2).flatten(1).cpu())
        labels.append(batch["labels_cls"])
        if bi % 20 == 0:
            print(f"batch {bi}/{len(dl)}")

    out = {
        "layers": layers,
        "labels": torch.cat(labels),
        **{f"layer{li}_{k}": torch.cat(v) for li, d in feats.items() for k, v in d.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {args.out}: {out['labels'].shape[0]} clips, layers {layers}")


if __name__ == "__main__":
    main()
