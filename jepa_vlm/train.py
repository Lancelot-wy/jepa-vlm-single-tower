"""Training entry point (Phase A regression / Phase B joint CE+regression).

Single GPU / multi-GPU via accelerate:
  python -m jepa_vlm.train --config configs/phase_a_v21.yaml [key=value overrides...]
  accelerate launch -m jepa_vlm.train --config configs/phase_a_v21.yaml

The loop is deliberately framework-light (plain torch + accelerate DDP) so it can be
ported onto the company multi-node harness by swapping the launcher only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data.datasets import ManifestVideoDataset, QAVideoDataset, QACollator, collate_visual
from .modeling.model import build_model


class ByteTokenizer:
    """UTF-8 byte tokenizer for tiny-config smoke tests (ids < 256 fit the tiny vocab)."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


def build_dataloaders(cfg: Config, tokenizer=None):
    tc, mc = cfg.train, cfg.model
    common = dict(
        data_root=tc.data_root, num_frames=tc.num_frames, sample_fps=tc.sample_fps,
        frame_sampling=tc.frame_sampling, frame_size=mc.frame_size,
        duplicate_frames=mc.duplicate_frames,
        # A fixed manifest index must receive the same temporal augmentation in
        # paired CE/MTP arms.  This is especially important for EXP-10, where
        # otherwise each arm would draw a different shuffle/reverse/offset from
        # OS entropy despite sharing a nominal seed.
        seed=tc.seed,
    )
    if tc.phase == "b":
        assert tokenizer is not None, "phase b needs a tokenizer"
        train_ds = QAVideoDataset(tc.text_manifest, min_flow=tc.min_flow, training=True,
                                  temporal_qa_ratio=tc.temporal_qa_ratio,
                                  temporal_qa_templates=tc.temporal_qa_templates, **common)
        tokens_per_clip = tc.num_frames * mc.tokens_per_frame
        # special-token ids are wired in from the model config in main()
        collate = QACollator(tokenizer, {}, tokens_per_clip, tc.max_text_len)
    else:
        train_ds = ManifestVideoDataset(tc.train_manifest, min_flow=tc.min_flow, training=True, **common)
        collate = collate_visual
    val_dl = None
    if tc.val_manifest:
        val_ds = ManifestVideoDataset(tc.val_manifest, training=False, **common)
        val_dl = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False,
                            num_workers=tc.num_workers, collate_fn=collate_visual, drop_last=True)
    train_dl = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                          num_workers=tc.num_workers, collate_fn=collate, drop_last=True,
                          persistent_workers=tc.num_workers > 0)
    return train_dl, val_dl


def make_optimizer(model, cfg: Config):
    tc = cfg.train
    new_keys = ("reg_head", "mtp_heads", "mask_embed", "attn_pool", "lora_")
    new_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (new_params if any(k in n for k in new_keys) else backbone_params).append(p)
    groups = [{"params": new_params, "lr": tc.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": tc.lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=tc.weight_decay, betas=(0.9, 0.95))


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def quick_eval(model, val_dl, cfg: Config, accelerator, max_batches: int):
    model.eval()
    agg: dict[str, list] = {}
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        out = model(
            pixel_values=batch["pixel_values"].to(accelerator.device, dtype=next(model.parameters()).dtype),
            grid_thw=batch["grid_thw"],
        )
        for k, v in out.metrics.items():
            agg.setdefault(k, []).append(v)
    model.train()
    m = {f"val/{k}": sum(v) / len(v) for k, v in agg.items() if v}
    if "val/copy_mse" in m and m["val/copy_mse"] > 0:
        # plan eval #3: masked regression loss vs copy-nearest-unmasked baseline.
        # ratio must be < 1 by a clear margin, otherwise the model found the trivial solution.
        m["val/nontrivial_ratio"] = m["val/reg_loss"] / m["val/copy_mse"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tc = cfg.train

    # find_unused_parameters=True: some variants leave params gradient-free (e.g. v1
    # never uses mask_embed; bidir/mtp-off skip the MTP heads), which otherwise trips
    # DDP's reduction check. Small overhead; harmless for the fully-used v2.1 runs.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=tc.grad_accum, kwargs_handlers=[ddp_kwargs]
    )
    set_seed(tc.seed)
    os.makedirs(tc.output_dir, exist_ok=True)

    tokenizer = None
    if tc.phase == "b":
        if cfg.model.tiny_config:
            tokenizer = ByteTokenizer()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained)

    model = build_model(cfg)
    train_dl, val_dl = build_dataloaders(cfg, tokenizer)
    if tc.phase == "b":  # wire real special-token ids into the collator
        ids = {k: getattr(model.hf_config, k) for k in
               ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
        train_dl.collate_fn.ids = ids

    optimizer = make_optimizer(model, cfg)

    # Do not pass the scheduler through Accelerator.prepare().  Accelerate's
    # distributed scheduler may advance once per process, while `max_steps` in
    # this project is explicitly an optimizer-update count.  Keeping a plain
    # scheduler also makes its state and the logged step unambiguous.
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, tc.warmup_steps, tc.max_steps))
    unwrapped = accelerator.unwrap_model(model)
    model_dtype = next(unwrapped.parameters()).dtype

    start_step = 0
    if tc.resume:
        state = torch.load(os.path.join(tc.resume, "state.pt"), map_location="cpu", weights_only=False)
        if state.get("step_unit") != "optimizer_update":
            raise RuntimeError(
                "Refusing to resume a legacy checkpoint whose `step` counted micro-batches. "
                "Start a fresh run or explicitly convert the checkpoint/scheduler state."
            )
        unwrapped.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = state["step"]
        accelerator.print(f"resumed from {tc.resume} @ step {start_step}")

    if accelerator.is_main_process:
        with open(os.path.join(tc.output_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    log_path = os.path.join(tc.output_dir, "log.jsonl")
    tb_writer = None
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(tc.output_dir, "tb"))

    def save(step):
        if not accelerator.is_main_process:
            return
        ckpt_dir = os.path.join(tc.output_dir, f"step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in unwrapped.state_dict().items() if k in trainable}
        torch.save({"model": sd, "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict(), "step": step,
                    "step_unit": "optimizer_update",
                    "config": cfg.to_dict()}, os.path.join(ckpt_dir, "state.pt"))
        accelerator.print(f"saved {ckpt_dir} ({len(sd)} trainable tensors)")

    model.train()
    step = start_step
    t0 = time.time()
    running: dict[str, float] = {}
    running_count = 0
    data_iter = iter(train_dl)
    while step < tc.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
        with accelerator.accumulate(model):
            kwargs = dict(
                pixel_values=batch["pixel_values"].to(accelerator.device, dtype=model_dtype),
                grid_thw=batch["grid_thw"],
            )
            if tc.phase == "b":
                kwargs.update(
                    input_ids=batch["input_ids"].to(accelerator.device),
                    attention_mask=batch["attention_mask"].to(accelerator.device),
                    labels=batch["labels"].to(accelerator.device),
                )
            out = model(**kwargs)
            accelerator.backward(out.loss)
            if accelerator.sync_gradients and tc.grad_clip > 0:
                accelerator.clip_grad_norm_(unwrapped.trainable_parameters(), tc.grad_clip)
            did_update = accelerator.sync_gradients
            optimizer.step()
            did_update = did_update and not optimizer.step_was_skipped
            if did_update:
                sched.step()
            optimizer.zero_grad(set_to_none=True)

        log_metrics = {"loss": float(out.loss.detach()), **out.metrics}
        if tc.phase == "b":
            log_metrics.update({
                "answer_tokens": float(batch["answer_token_count"].float().mean()),
                "answer_truncated_frac": float(batch["answer_truncated"].float().mean()),
                "question_truncated_frac": float(batch["question_truncated"].float().mean()),
            })
        for k, v in log_metrics.items():
            running[k] = running.get(k, 0.0) + v
        running_count += 1
        if not did_update:
            continue
        step += 1

        if step % tc.log_every == 0:
            metric_count = running_count
            rec = {"step": step, "lr": sched.get_last_lr()[0],
                   "sec_per_step": (time.time() - t0) / tc.log_every,
                   **{k: v / metric_count for k, v in running.items()}}
            running, running_count, t0 = {}, 0, time.time()
            accelerator.print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                          for k, v in rec.items()}))
            if accelerator.is_main_process:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if tb_writer is not None:
                    for k, v in rec.items():
                        if k != "step" and isinstance(v, (int, float)):
                            tb_writer.add_scalar(f"train/{k}", v, step)

        if val_dl is not None and step % tc.eval_every == 0:
            m = quick_eval(unwrapped, val_dl, cfg, accelerator, tc.eval_batches)
            accelerator.print(json.dumps({"step": step, **{k: round(v, 5) for k, v in m.items()}}))
            if accelerator.is_main_process:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, **m}) + "\n")
                if tb_writer is not None:
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f"val/{k}", v, step)

        if step % tc.save_every == 0:
            save(step)

    save(step)
    if tb_writer is not None:
        tb_writer.close()
    accelerator.print("done")


if __name__ == "__main__":
    main()
