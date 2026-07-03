"""Experiment configuration.

All variants / ablations are controlled from YAML files (configs/*.yaml).
A YAML file may set `_base_: other.yaml` to inherit and override.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict

import yaml


@dataclass
class ModelConfig:
    # HF checkpoint or path. For local smoke tests use tiny_config=true instead.
    pretrained: str = "Qwen/Qwen3-VL-2B-Instruct"
    tiny_config: bool = False          # random-init tiny model for CPU smoke tests
    dtype: str = "bfloat16"            # bfloat16 | float32
    attn_implementation: str = "sdpa"  # sdpa | flash_attention_2 | eager

    # --- latent frame layout ---
    frame_size: int = 256              # square input frames, multiple of 32
    tokens_per_frame: int = 4          # pooled tokens per latent frame (2x2)
    pooling: str = "avg"               # avg | attn  (attn = learned 4-query attention pooling)
    duplicate_frames: bool = True      # duplicate each sampled frame x2 so that one latent
                                       # slot == one sampled frame (temporal_patch_size=2)

    # --- masking ---
    mask_variant: str = "v2.1"         # v1 (no mask) | v2.1 (loss on [M] only) | v2.2 (loss everywhere)
    mask_ratio: float = 0.5
    mask_mode: str = "tube"            # tube = whole latent frames (contiguous runs or scattered)
                                       # patch = random token-level mask (negative-control ablation)
    mask_tube_max_run: int = 4         # max contiguous run length when sampling tube masks

    # --- heads ---
    reg_head_hidden: int = 0           # 0 -> use LLM hidden size
    mtp_enabled: bool = True
    mtp_k: int = 4                     # predict h_{t+1..t+k}

    # --- deepstack ---
    use_deepstack: bool = True         # pool & inject deepstack features (zeroed at masked slots)

    # --- attention over visual tokens ---
    bidirectional_visual: bool = False # ablation: visual tokens attend bidirectionally (disables MTP)

    # --- probes ---
    probe_layers: tuple = (-1,)        # relative layer indices for hidden-state probes; resolved at runtime
                                       # default set in train/probe scripts: middle + last


@dataclass
class TrainConfig:
    seed: int = 0
    train_manifest: str = ""
    val_manifest: str = ""
    data_root: str = ""                # prepended to relative video paths in manifests
    min_flow: float = 0.0              # drop clips whose manifest `flow` < min_flow (0 = keep all)

    num_frames: int = 16               # sampled frames T (= latent slots when duplicate_frames)
    sample_fps: float = 2.0            # target sampling fps; falls back to uniform if clip too short
    frame_sampling: str = "fps_or_uniform"  # fps_or_uniform | uniform

    batch_size: int = 8                # per device
    num_workers: int = 4
    lr: float = 1e-4                   # heads / new params
    lr_backbone: float = 1e-5          # vision encoder + LLM
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0
    grad_accum: int = 1
    log_every: int = 20
    save_every: int = 1000
    eval_every: int = 1000             # quick val: reg loss + non-triviality ratio + collapse monitor
    eval_batches: int = 50
    output_dir: str = "runs/default"
    resume: str = ""                   # checkpoint dir to resume from

    gradient_checkpointing: bool = True

    # --- trainable parts (Phase A: freeze lm_head & embed_tokens implicitly: they receive no grads) ---
    train_vision: bool = True
    train_llm: str = "full"            # full | lora | frozen
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # --- Phase B (joint CE + regression) ---
    phase: str = "a"                   # a | b
    lambda_reg: float = 0.2            # L = L_CE + lambda * L_reg   (phase b only)
    text_manifest: str = ""            # phase b QA manifest (jsonl with video/question/answer)
    max_text_len: int = 256


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml_with_base(path: str) -> dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base_rel = d.pop("_base_", None)
    if base_rel:
        base = _load_yaml_with_base(os.path.join(os.path.dirname(path), base_rel))
        d = _deep_update(base, d)
    return d


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    """Load YAML config; `overrides` are dotted key=value strings, e.g. model.mask_ratio=0.75."""
    d = _load_yaml_with_base(path)
    cfg = Config()
    raw = cfg.to_dict()
    _deep_update(raw, d)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        old = node.get(parts[-1])
        node[parts[-1]] = yaml.safe_load(val)
        if old is not None and not isinstance(node[parts[-1]], type(old)) and not (
            isinstance(old, float) and isinstance(node[parts[-1]], int)
        ):
            node[parts[-1]] = type(old)(node[parts[-1]])
    known_m = {f for f in ModelConfig.__dataclass_fields__}
    known_t = {f for f in TrainConfig.__dataclass_fields__}
    unknown = (set(raw["model"]) - known_m) | (set(raw["train"]) - known_t)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(model=ModelConfig(**raw["model"]), train=TrainConfig(**raw["train"]))
    if cfg.model.frame_size % 32 != 0:
        raise ValueError("model.frame_size must be a multiple of 32 (patch 16 x merge 2)")
    if cfg.model.mask_variant not in ("v1", "v2.1", "v2.2"):
        raise ValueError(f"bad mask_variant {cfg.model.mask_variant}")
    return cfg
