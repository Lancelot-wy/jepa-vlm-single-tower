"""Training entry point (Phase A regression / Phase B joint CE+regression).

Single GPU / multi-GPU via accelerate:
  python -m jepa_vlm.train --config configs/phase_a_v21.yaml [key=value overrides...]
  accelerate launch -m jepa_vlm.train --config configs/phase_a_v21.yaml

The loop is deliberately framework-light (plain torch + accelerate DDP) so it can be
ported onto the company multi-node harness by swapping the launcher only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate import skip_first_batches
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from .config import (
    Config,
    is_exp12_config,
    load_config,
    resolved_raw_num_frames,
    resolved_temporal_units,
    resolved_visual_tokens,
)
from .data.datasets import ManifestVideoDataset, QAVideoDataset, QACollator, collate_visual
from .data.event_dataset import EventCollator, EventVideoDataset
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
        data_root=tc.data_root, num_frames=resolved_raw_num_frames(cfg), sample_fps=tc.sample_fps,
        frame_sampling=tc.frame_sampling, frame_size=mc.frame_size,
        duplicate_frames=mc.duplicate_frames, temporal_patch_size=tc.temporal_patch_size,
        state_horizon_units=tc.state_horizon_units,
        # A fixed manifest index must receive the same temporal augmentation in
        # paired CE/MTP arms.  This is especially important for EXP-10, where
        # otherwise each arm would draw a different shuffle/reverse/offset from
        # OS entropy despite sharing a nominal seed.
        seed=tc.seed,
        deterministic_order=tc.deterministic_data_order,
    )
    if tc.phase == "b":
        assert tokenizer is not None, "phase b needs a tokenizer"
        train_ds = QAVideoDataset(tc.text_manifest, min_flow=tc.min_flow, training=True,
                                  temporal_qa_ratio=tc.temporal_qa_ratio,
                                  temporal_qa_templates=tc.temporal_qa_templates, **common)
        tokens_per_clip = resolved_temporal_units(cfg) * resolved_visual_tokens(cfg)
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
    train_dl = DataLoader(train_ds, batch_size=tc.batch_size,
                          shuffle=not tc.deterministic_data_order,
                          num_workers=tc.num_workers, collate_fn=collate, drop_last=True,
                          persistent_workers=tc.num_workers > 0)
    event_dl = None
    if mc.event_condition_enable:
        if tokenizer is None:
            raise ValueError("event-conditioned training needs the Phase-B tokenizer")
        if not tc.event_dataset_path:
            raise ValueError("event_condition_enable requires train.event_dataset_path")
        event_ds = EventVideoDataset(
            tc.event_dataset_path, split="train",
            raw_num_frames=resolved_raw_num_frames(cfg), sample_fps=tc.sample_fps,
            frame_sampling=tc.frame_sampling, frame_size=mc.frame_size,
            temporal_patch_size=tc.temporal_patch_size, seed=tc.seed,
            inner_min=mc.event_target_inner_min, inner_max=mc.event_target_inner_max,
            direction_mode=mc.event_direction_mode,
        )
        event_dl = DataLoader(
            event_ds, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers,
            collate_fn=EventCollator(tokenizer), drop_last=True,
            persistent_workers=tc.num_workers > 0,
        )
    return train_dl, val_dl, event_dl


def make_optimizer(model, cfg: Config):
    tc = cfg.train
    legacy_new_keys = (
        "reg_head", "mtp_heads", "mask_embed", "attn_pool", "lora_",
        "orca_queries", "orca_head",
    )
    state_keys = (
        "state_query_builder", "event_query_builder", "state_transition_head",
        "event_direction_embedding",
    )
    legacy_params, state_params, backbone_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in state_keys):
            state_params.append(p)
        elif any(k in n for k in legacy_new_keys):
            legacy_params.append(p)
        else:
            backbone_params.append(p)
    groups = []
    if legacy_params:
        groups.append({"name": "legacy_new", "params": legacy_params, "lr": tc.lr})
    if state_params:
        groups.append({
            "name": "state_query_head", "params": state_params,
            "lr": tc.state_head_learning_rate,
        })
    if backbone_params:
        groups.append({
            "name": "base_model", "params": backbone_params,
            "lr": tc.base_model_learning_rate or tc.lr_backbone,
        })
    if not groups:
        raise ValueError("optimizer has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=tc.weight_decay, betas=(0.9, 0.95))
    if is_exp12_config(cfg):
        visual_ids = {id(parameter) for parameter in model.visual.parameters()}
        leaked = sum(
            parameter.numel() for group in optimizer.param_groups for parameter in group["params"]
            if id(parameter) in visual_ids
        )
        if leaked:
            raise RuntimeError(f"frozen visual/merger parameters entered optimizer: {leaked}")
    return optimizer


def parameter_audit(model, optimizer, cfg: Config) -> dict:
    named = list(model.named_parameters())
    visual = list(model.visual.named_parameters())
    merger = [(name, value) for name, value in visual if "merger" in name.lower()]
    vit = [(name, value) for name, value in visual if "merger" not in name.lower()]
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    query_names = ("state_query_builder", "event_query_builder")
    head_names = ("state_transition_head", "event_direction_embedding")
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "name": group.get("name", "unnamed"),
            "lr": float(group["lr"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "tensors": len(group["params"]),
        })
    return {
        "total_parameters": sum(value.numel() for _, value in named),
        "trainable_parameters": sum(value.numel() for _, value in named if value.requires_grad),
        "frozen_vit_parameters": sum(value.numel() for _, value in vit if not value.requires_grad),
        "frozen_merger_parameters": sum(value.numel() for _, value in merger if not value.requires_grad),
        "trainable_llm_parameters": sum(
            value.numel() for name, value in named
            if name.startswith("language_model.") and value.requires_grad
        ),
        "query_parameters": sum(
            value.numel() for name, value in named
            if any(key in name for key in query_names) and value.requires_grad
        ),
        "head_parameters": sum(
            value.numel() for name, value in named
            if any(key in name for key in head_names) and value.requires_grad
        ),
        "optimizer_parameter_groups": groups,
        "frozen_parameters_in_optimizer": sum(
            value.numel() for _, value in named
            if not value.requires_grad and id(value) in optimizer_ids
        ),
        "visual_parameters_in_optimizer": sum(
            value.numel() for _, value in visual if id(value) in optimizer_ids
        ),
        "physical_visual_module_count": 1,
        "state_predictor_mode": cfg.model.state_predictor_mode,
    }


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _environment_audit() -> dict:
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "hostname": platform.node(),
    }


def _resource_audit(cfg: Config) -> dict:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "worker_environment": {
            key: os.environ.get(key) for key in (
                "TF_CONFIG", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK",
                "CUDA_VISIBLE_DEVICES",
            ) if os.environ.get(key) is not None
        },
        "world_size": world,
        "per_device_train_batch_size": cfg.train.batch_size,
        "gradient_accumulation_steps": cfg.train.grad_accum,
        "effective_batch_size": world * cfg.train.batch_size * cfg.train.grad_accum,
        "visible_gpu_count": torch.cuda.device_count(),
    }


def _rng_state(event_rng: random.Random) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "event_rng": event_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, event_rng: random.Random) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    event_rng.setstate(state["event_rng"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_event_batch(batch: dict, device, dtype) -> dict:
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif "pixel_values" in key:
            moved[key] = value.to(device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


def checkpoint_dir(output_dir: str, style: str, step: int) -> str:
    return os.path.join(output_dir, f"checkpoint-{step}" if style == "checkpoint" else f"step_{step}")


def _resume_contract(config: dict) -> dict:
    """Fields that may not change inside one scientific training arm."""
    value = json.loads(json.dumps(config))
    train = value["train"]
    for key in (
        "output_dir", "resume", "max_steps", "save_every", "eval_every",
        "log_every", "num_workers", "checkpoint_style", "log_filename",
    ):
        train.pop(key, None)
    return value


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
    train_dl, val_dl, event_dl = build_dataloaders(cfg, tokenizer)
    if tc.phase == "b":  # wire real special-token ids into the collator
        ids = {k: getattr(model.hf_config, k) for k in
               ("video_token_id", "vision_start_token_id", "vision_end_token_id")}
        train_dl.collate_fn.ids = ids

    optimizer = make_optimizer(model, cfg)
    audit = parameter_audit(model, optimizer, cfg)
    if audit["frozen_parameters_in_optimizer"] or audit["visual_parameters_in_optimizer"]:
        raise RuntimeError(f"invalid optimizer parameter audit: {audit}")
    if is_exp12_config(cfg):
        model.assert_exp12_frozen_visual()

    if accelerator.is_main_process:
        with open(os.path.join(tc.output_dir, "config_used.yaml"), "w") as handle:
            yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
        with open(os.path.join(tc.output_dir, "config.json"), "w") as handle:
            json.dump(cfg.to_dict(), handle, indent=2)
        with open(os.path.join(tc.output_dir, "parameter_audit.json"), "w") as handle:
            json.dump(audit, handle, indent=2)
        with open(os.path.join(tc.output_dir, "environment.json"), "w") as handle:
            json.dump(_environment_audit(), handle, indent=2)
        with open(os.path.join(tc.output_dir, "resource_audit.json"), "w") as handle:
            json.dump(_resource_audit(cfg), handle, indent=2)
        with open(os.path.join(tc.output_dir, "git_commit.txt"), "w") as handle:
            handle.write(_git_commit() + "\n")
        manifest = tc.text_manifest if tc.phase == "b" else tc.train_manifest
        if manifest and os.path.isfile(manifest):
            with open(os.path.join(tc.output_dir, "manifest.sha256"), "w") as handle:
                handle.write(_sha256(manifest) + "\n")

    # Do not pass the scheduler through Accelerator.prepare().  Accelerate's
    # distributed scheduler may advance once per process, while `max_steps` in
    # this project is explicitly an optimizer-update count.  Keeping a plain
    # scheduler also makes its state and the logged step unambiguous.
    if event_dl is None:
        model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    else:
        model, optimizer, train_dl, event_dl = accelerator.prepare(
            model, optimizer, train_dl, event_dl
        )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, tc.warmup_steps, tc.max_steps))
    unwrapped = accelerator.unwrap_model(model)
    model_dtype = next(unwrapped.parameters()).dtype

    start_step = 0
    data_batches_seen = 0
    event_batches_seen = 0
    event_rng = random.Random(tc.seed + 120012)
    if tc.resume:
        state = torch.load(os.path.join(tc.resume, "state.pt"), map_location="cpu", weights_only=False)
        if state.get("step_unit") != "optimizer_update":
            raise RuntimeError(
                "Refusing to resume a legacy checkpoint whose `step` counted micro-batches. "
                "Start a fresh run or explicitly convert the checkpoint/scheduler state."
            )
        saved_config = state.get("config")
        if is_exp12_config(cfg):
            if saved_config is None:
                raise RuntimeError("EXP-12 resume checkpoint is missing its config contract")
            if _resume_contract(saved_config) != _resume_contract(cfg.to_dict()):
                raise RuntimeError(
                    "EXP-12 resume changed a scientific config field; start a new arm instead"
                )
        missing, unexpected = unwrapped.load_state_dict(state["model"], strict=False)
        if is_exp12_config(cfg):
            trainable_names = {
                name for name, parameter in unwrapped.named_parameters()
                if parameter.requires_grad
            }
            missing_trainable = sorted(set(missing) & trainable_names)
            if missing_trainable or unexpected:
                raise RuntimeError(
                    "EXP-12 checkpoint/model mismatch: "
                    f"missing trainable={missing_trainable[:8]}, unexpected={unexpected[:8]}"
                )
        if is_exp12_config(cfg) and cfg.model.state_predictor_mode != "none":
            unwrapped.load_checkpoint_aux_state(state.get("model_aux", {}))
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = state["step"]
        data_batches_seen = int(state.get("data_batches_seen", start_step * tc.grad_accum))
        event_batches_seen = int(state.get("event_batches_seen", 0))
        if is_exp12_config(cfg):
            if "rng_state" not in state:
                raise RuntimeError("EXP-12 resume checkpoint is missing RNG state")
            _restore_rng_state(state["rng_state"], event_rng)
        accelerator.print(f"resumed from {tc.resume} @ step {start_step}")

    log_path = os.path.join(tc.output_dir, tc.log_filename)
    tb_writer = None
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(tc.output_dir, "tb"))

    def save(step):
        if not accelerator.is_main_process:
            return
        ckpt_dir = checkpoint_dir(tc.output_dir, tc.checkpoint_style, step)
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in unwrapped.state_dict().items() if k in trainable}
        state_path = os.path.join(ckpt_dir, "state.pt")
        tmp_path = f"{state_path}.tmp"
        # A shared JuiceFS write can be interrupted when an allocation is
        # reclaimed. Publish a checkpoint only after its full payload flushes,
        # so resume can safely select the newest valid state.
        with open(tmp_path, "wb") as f:
            torch.save({"model": sd, "model_aux": unwrapped.checkpoint_aux_state(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": sched.state_dict(), "step": step,
                        "step_unit": "optimizer_update",
                        "data_batches_seen": data_batches_seen,
                        "event_batches_seen": event_batches_seen,
                        "rng_state": _rng_state(event_rng),
                        "config": cfg.to_dict()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
        meta_path = os.path.join(ckpt_dir, "checkpoint_meta.json")
        meta_tmp = f"{meta_path}.tmp"
        with open(meta_tmp, "w") as handle:
            json.dump({
                "step": step,
                "step_unit": "optimizer_update",
                "state_file": "state.pt",
                "state_bytes": os.path.getsize(state_path),
                "git_commit": _git_commit(),
            }, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(meta_tmp, meta_path)
        accelerator.print(f"saved {ckpt_dir} ({len(sd)} trainable tensors)")

    model.train()
    step = start_step
    t0 = time.time()
    last_log_step = start_step
    running: dict[str, float] = {}
    running_count = 0
    if tc.deterministic_data_order and data_batches_seen:
        train_dl = skip_first_batches(train_dl, data_batches_seen % len(train_dl))
    if event_dl is not None and event_batches_seen:
        event_dl = skip_first_batches(event_dl, event_batches_seen % len(event_dl))
    data_iter = iter(train_dl)
    event_iter = iter(event_dl) if event_dl is not None else None
    while step < tc.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
        data_batches_seen += 1
        event_batch = None
        if event_iter is not None and event_rng.random() < cfg.model.event_aux_probability:
            try:
                event_batch = next(event_iter)
            except StopIteration:
                event_iter = iter(event_dl)
                event_batch = next(event_iter)
            event_batches_seen += 1
            event_batch = _move_event_batch(event_batch, accelerator.device, model_dtype)
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
                    state_eligible=batch.get("state_eligible", None).to(accelerator.device)
                    if batch.get("state_eligible", None) is not None else None,
                    event_batch=event_batch,
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
        if "video_stats" in batch:
            video = batch["video_stats"]
            log_metrics.update({
                "video/raw_frame_count": float(video["raw_frame_count"].float().mean()),
                "video/unique_frame_count": float(video["unique_frame_count"].float().mean()),
                "video/temporal_unit_count": float(video["temporal_unit_count"].float().mean()),
                "video/duplicate_adjacent_ratio": float(
                    video["duplicate_adjacent_ratio"].float().mean()
                ),
                "video/effective_fps": float(video["effective_fps"].float().mean()),
                "video/state_eligible_fraction": float(
                    batch.get("state_eligible", torch.zeros(1)).float().mean()
                ),
                "video/state_skipped_short_fraction": float(
                    video["state_skipped_short"].float().mean()
                ),
                "video/state_skipped_temporal_augmentation_fraction": float(
                    video["state_skipped_temporal_augmentation"].float().mean()
                ),
                "video/decode_retry_count": float(
                    video["decode_retry_count"].float().mean()
                ),
                "video/decode_exception_fraction": float(
                    (video["decode_retry_count"] > 0).float().mean()
                ),
                "video/nearest_frame_substitution_fraction": float(
                    (
                        video["nearest_frame_substitutions"].float()
                        / video["raw_frame_count"].float().clamp_min(1)
                    ).mean()
                ),
            })
        non_finite = [
            key for key, value in log_metrics.items()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        ]
        if non_finite:
            raise FloatingPointError(f"non-finite EXP training metrics: {non_finite}")
        for k, v in log_metrics.items():
            running[k] = running.get(k, 0.0) + v
        running_count += 1
        if not did_update:
            continue
        step += 1

        if step % tc.log_every == 0:
            keys = sorted(running)
            packed = torch.tensor(
                [running[key] for key in keys] + [float(running_count)],
                dtype=torch.float32,
                device=accelerator.device,
            )
            packed = accelerator.reduce(packed, reduction="sum")
            metric_count = float(packed[-1].clamp_min(1.0).item())
            interval_steps = max(step - last_log_step, 1)
            elapsed = torch.tensor(time.time() - t0, device=accelerator.device)
            max_memory = torch.tensor(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                dtype=torch.float32,
                device=accelerator.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.MAX)
            rec = {"step": step, "lr": sched.get_last_lr()[0],
                   "interval_steps": interval_steps,
                   "sec_per_step": float(elapsed.item()) / interval_steps,
                   **{key: float(packed[index].item()) / metric_count
                      for index, key in enumerate(keys)}}
            rec["samples_per_sec"] = (
                tc.batch_size * accelerator.num_processes * tc.grad_accum /
                max(rec["sec_per_step"], 1e-8)
            )
            rec["max_memory_gb"] = float(max_memory.item()) / (1024 ** 3)
            for index, group in enumerate(optimizer.param_groups):
                rec[f"lr/{group.get('name', index)}"] = float(group["lr"])
            running, running_count, t0 = {}, 0, time.time()
            last_log_step = step
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
