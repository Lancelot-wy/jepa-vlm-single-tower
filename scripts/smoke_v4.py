"""CPU smoke test for the V4 arms: one forward+backward per representative config
(tiny random-init model, fake video + QA batch). Verifies the dual_view / reg_enabled
code paths produce finite losses and gradients before burning cluster GPUs.

  .venv/bin/python scripts/smoke_v4.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import torch

from jepa_vlm.config import load_config
from jepa_vlm.data.datasets import QACollator
from jepa_vlm.data.video_io import patchify, resize_center_crop
from jepa_vlm.modeling.model import build_model
from jepa_vlm.train import ByteTokenizer

ARMS = ["v4_ctrl_s0", "v4_dv25_s0", "v4_mtp1", "v4_dv25_mtp1", "v4_dvce25"]
OVERRIDES = ["model.tiny_config=true", "model.dtype=float32", "model.frame_size=64",
             "train.num_frames=8", "train.max_text_len=192"]

for arm in ARMS:
    cfg = load_config(f"configs/{arm}.yaml", OVERRIDES)
    torch.manual_seed(0)
    model = build_model(cfg)
    model.train()

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(cfg.train.num_frames, 96, 96, 3), dtype=np.uint8)
    pv, grid = patchify(resize_center_crop(frames, cfg.model.frame_size),
                        cfg.model.duplicate_frames)
    tok = ByteTokenizer()
    ids = {k: getattr(model.hf_config, k) for k in
           ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
    collator = QACollator(tok, ids, cfg.train.num_frames * cfg.model.tokens_per_frame,
                          cfg.train.max_text_len)
    batch = collator([
        {"pixel_values": pv, "grid_thw": grid, "question": "What happens?", "answer": "A. x"},
        {"pixel_values": pv, "grid_thw": grid, "question": "What happens?", "answer": "B. y"},
    ])
    out = model(
        pixel_values=batch["pixel_values"].float(),
        grid_thw=batch["grid_thw"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert out.loss is not None and torch.isfinite(out.loss), f"{arm}: bad loss {out.loss}"
    out.loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    parts = {k: (round(float(v), 4) if isinstance(v, torch.Tensor) else v)
             for k, v in [("loss", out.loss), ("ce", out.ce_loss),
                          ("reg", out.reg_loss), ("mtp", out.mtp_loss)] if v is not None}
    print(f"{arm:16s} OK  {parts}  grad_tensors={grads}")

print("\nSMOKE V4: ALL ARMS PASS")
