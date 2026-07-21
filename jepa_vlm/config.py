"""Experiment configuration.

All variants / ablations are controlled from YAML files (configs/*.yaml).
A YAML file may set `_base_: other.yaml` to inherit and override.
"""

from __future__ import annotations

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

    # --- EXP-12 real temporal units / visual-token sweep ---
    # A zero value keeps the historical `tokens_per_frame` contract.  EXP-12
    # configs set this explicitly to 4, 16, or 64.
    visual_tokens_per_unit: int = 0
    visual_pooling_mode: str = "spatial_avg"

    # --- masking ---
    mask_variant: str = "v2.1"         # v1 (no mask) | v2.1 (loss on [M] only) | v2.2 (loss everywhere)
    mask_ratio: float = 0.5
    mask_mode: str = "tube"            # tube = whole latent frames (contiguous runs or scattered)
                                       # patch = random token-level mask (negative-control ablation)
    mask_tube_max_run: int = 4         # max contiguous run length when sampling tube masks

    # --- heads ---
    reg_head_hidden: int = 0           # 0 -> use LLM hidden size
    reg_enabled: bool = True           # false -> no masked-reg loss (e.g. CE + MTP-only arms)
    mtp_enabled: bool = True
    mtp_k: int = 4                     # predict h_{t+1..t+k}

    # --- minimal Orca-inspired observation transition ---
    # This is intentionally narrower than the full Orca recipe: VQA/CE remains the
    # main task, while a frozen per-frame visual teacher supervises learnable query
    # tokens that predict a future visual state.  Event-conditioned training is a
    # later experiment once timestamped event manifests have been audited.
    orca_enabled: bool = False
    orca_use_queries: bool = True       # false = matched no-query ablation
    orca_query_tokens: int = 4          # one predictive query per pooled target token
    orca_target_gap: int = 1            # sampled-frame gap; at 2 fps, gap=2 is one second

    # --- EXP-12 single-tower state prediction ---
    # `none` is pure CE.  The three predictive modes share the same frozen
    # visual states and transition-head specification.
    state_predictor_mode: str = "none"  # none | no_query | query | observation_event_query
    state_query_count: int | str = "auto"
    state_query_position_encoding: bool = True
    num_horizon_embeddings: int = 8
    state_horizon_values_seconds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    state_head_hidden: int = 0
    state_center_momentum: float = 0.99
    state_loss_weight: float = 0.05
    state_dynamic_threshold: float = 0.05
    state_dynamic_weighting: bool = True
    beat_copy_loss_weight: float = 0.0
    beat_copy_margin: float = 0.05
    random_mask_ratio: float = 0.0

    # Event code is implemented for the second sweep but remains disabled in
    # A0-A5.  Ordinary caption rows are never interpreted as events.
    event_condition_enable: bool = False
    event_loss_weight: float = 0.05
    event_aux_probability: float = 0.25
    event_direction_mode: str = "bidirectional"
    event_dynamic_threshold: float = 0.08
    event_query_count: int | str = "auto"
    event_target_sampling: str = "random_inner"
    event_target_inner_min: float = 0.2
    event_target_inner_max: float = 0.8
    event_hard_negative_mode: str = "same_video_other_event"

    # --- dual-view training (V4 validation round) ---
    # off: single forward (round-3 behaviour: CE and reg share the masked input)
    # reg: clean view -> CE (input never sees [M]); masked view -> visual-only reg(+mtp).
    #      L = CE_clean + lambda_reg * L_reg_masked
    # ce : both views run CE, no regression (isolates the consistency/regularizer effect).
    #      L = 0.5 * (CE_clean + CE_masked)
    dual_view: str = "off"             # off | reg | ce

    # --- deepstack ---
    use_deepstack: bool = True         # pool & inject deepstack features (zeroed at masked slots)

    # --- attention over visual tokens ---
    bidirectional_visual: bool = False # ablation: visual tokens attend bidirectionally (disables MTP)

    # --- mechanism-fix arms (added after the first cluster round, see results/ANALYSIS) ---
    var_reg_weight: float = 0.0        # VICReg-style per-dim std hinge on the (normed, non-detached)
                                       # online features; counters directional collapse. 0 = off.
    var_reg_gamma: float = 0.4         # std floor; calibrate to the frozen-ViT target_std level
    residual_target: bool = False      # regress h_t - h_{nearest visible frame} instead of h_t:
                                       # the copy solution becomes exactly "predict zero", so the
                                       # nontrivial ratio (reg/copy) must go below 1 to mean anything.
                                       # Requires v2.1 + tube mask + mtp_enabled=false.

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

    # EXP-12 separates decoded images from Qwen temporal units.  Zero values
    # retain the legacy `num_frames` interpretation for old experiments.
    raw_num_frames: int = 0
    clip_duration_seconds: float = 0.0
    temporal_patch_size: int = 2
    num_temporal_units: int = 0
    state_horizon_units: int = 2
    state_horizon_seconds: float = 1.0
    deterministic_data_order: bool = False

    batch_size: int = 8                # per device
    num_workers: int = 4
    lr: float = 1e-4                   # heads / new params
    lr_backbone: float = 1e-5          # vision encoder + LLM
    base_model_learning_rate: float = 0.0  # 0 -> use lr_backbone (legacy compatible)
    state_head_learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 20000
    lr_scheduler_type: str = "cosine"
    grad_clip: float = 1.0
    grad_accum: int = 1
    log_every: int = 20
    save_every: int = 1000
    eval_every: int = 1000             # quick val: reg loss + non-triviality ratio + collapse monitor
    eval_batches: int = 50
    output_dir: str = "runs/default"
    resume: str = ""                   # checkpoint dir to resume from
    checkpoint_style: str = "step"     # step -> step_N; checkpoint -> checkpoint-N
    log_filename: str = "log.jsonl"

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
    temporal_qa_ratio: float = 0.3     # phase b: fraction of samples replaced by on-the-fly
                                       # temporal-order QA (shuffle/reverse, yes/no)
    temporal_qa_templates: str = "v1"  # v1 = 帧序是非题（EXP-04/08 行为）；v2 = 5 模板族
                                       # (order_yn/order_mcq/playback/speed/pan)，对症
                                       # TempCompass 的 direction/speed/order 弱项（EXP-09）

    # Separate audited event manifest.  It is read only when
    # `model.event_condition_enable=true`.
    event_dataset_path: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def resolved_visual_tokens(cfg: Config) -> int:
    """Return K while preserving old `tokens_per_frame` configs."""
    return cfg.model.visual_tokens_per_unit or cfg.model.tokens_per_frame


def resolved_raw_num_frames(cfg: Config) -> int:
    """Number of decoded images before temporal patchification."""
    return cfg.train.raw_num_frames or cfg.train.num_frames


def resolved_temporal_units(cfg: Config) -> int:
    """Number of temporal patches presented to the visual merger."""
    if cfg.train.num_temporal_units:
        return cfg.train.num_temporal_units
    if cfg.model.duplicate_frames:
        return cfg.train.num_frames
    return (resolved_raw_num_frames(cfg) + cfg.train.temporal_patch_size - 1) // cfg.train.temporal_patch_size


def resolved_query_count(cfg: Config, *, event: bool = False) -> int:
    value = cfg.model.event_query_count if event else cfg.model.state_query_count
    return resolved_visual_tokens(cfg) if value == "auto" else int(value)


def is_exp12_config(cfg: Config) -> bool:
    return cfg.model.visual_tokens_per_unit > 0 or cfg.model.state_predictor_mode != "none"


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
    if cfg.model.dual_view not in ("off", "reg", "ce"):
        raise ValueError(f"bad dual_view {cfg.model.dual_view}")
    if cfg.model.dual_view != "off" and cfg.model.mask_variant == "v1":
        raise ValueError("dual_view needs a mask variant (v2.1/v2.2); v1 has no masked view")
    if cfg.model.visual_pooling_mode != "spatial_avg":
        raise ValueError("visual_pooling_mode must be spatial_avg")
    if cfg.model.visual_tokens_per_unit and cfg.model.visual_tokens_per_unit not in (4, 16, 64):
        raise ValueError("visual_tokens_per_unit must be one of 4, 16, 64")
    if cfg.train.temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    if cfg.train.sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if cfg.train.lr_scheduler_type != "cosine":
        raise ValueError("only lr_scheduler_type=cosine is supported")
    if cfg.train.checkpoint_style not in ("step", "checkpoint"):
        raise ValueError("checkpoint_style must be step or checkpoint")
    if cfg.model.state_predictor_mode not in (
        "none", "no_query", "query", "observation_event_query"
    ):
        raise ValueError(f"bad state_predictor_mode {cfg.model.state_predictor_mode}")
    if not 0.0 <= cfg.model.state_center_momentum < 1.0:
        raise ValueError("state_center_momentum must be in [0, 1)")
    if cfg.model.state_loss_weight < 0 or cfg.model.event_loss_weight < 0:
        raise ValueError("state/event loss weights must be nonnegative")
    if not 0.0 <= cfg.model.event_aux_probability <= 1.0:
        raise ValueError("event_aux_probability must be in [0, 1]")
    if cfg.model.state_dynamic_threshold <= 0 or cfg.model.event_dynamic_threshold <= 0:
        raise ValueError("dynamic thresholds must be positive")
    if cfg.model.random_mask_ratio != 0 and is_exp12_config(cfg):
        raise ValueError("EXP-12 does not use random masking")
    if cfg.model.num_horizon_embeddings < len(cfg.model.state_horizon_values_seconds):
        raise ValueError("num_horizon_embeddings is smaller than state_horizon_values_seconds")
    if (
        any(value <= 0 for value in cfg.model.state_horizon_values_seconds)
        or len(set(cfg.model.state_horizon_values_seconds))
        != len(cfg.model.state_horizon_values_seconds)
    ):
        raise ValueError("state_horizon_values_seconds must be positive and unique")
    if cfg.model.event_direction_mode not in ("forward", "backward", "bidirectional"):
        raise ValueError("bad event_direction_mode")
    if cfg.model.event_target_sampling != "random_inner":
        raise ValueError("only event_target_sampling=random_inner is supported")
    if not 0 <= cfg.model.event_target_inner_min < cfg.model.event_target_inner_max <= 1:
        raise ValueError("event target inner range must satisfy 0 <= min < max <= 1")
    if cfg.model.event_hard_negative_mode != "same_video_other_event":
        raise ValueError("only same_video_other_event hard negatives are supported")
    if cfg.model.orca_enabled:
        if cfg.model.mask_variant != "v1":
            raise ValueError("orca_enabled currently requires mask_variant=v1; test masking in a separate arm")
        if cfg.model.mtp_enabled or cfg.model.reg_enabled:
            raise ValueError("orca_enabled is an isolated transition objective; disable legacy reg and MTP")
        if cfg.model.orca_use_queries and cfg.model.orca_query_tokens != cfg.model.tokens_per_frame:
            raise ValueError("orca_query_tokens must equal tokens_per_frame in the pilot")
        if cfg.model.orca_target_gap < 1 or cfg.model.orca_target_gap >= cfg.train.num_frames:
            raise ValueError("orca_target_gap must be in [1, num_frames)")
        if cfg.train.train_vision:
            raise ValueError("orca_enabled requires train_vision=false so the target encoder is frozen")
    if cfg.model.orca_enabled and cfg.model.state_predictor_mode != "none":
        raise ValueError("legacy orca_enabled and EXP-12 state_predictor_mode cannot be combined")
    if is_exp12_config(cfg):
        raw_frames = resolved_raw_num_frames(cfg)
        units = resolved_temporal_units(cfg)
        k = resolved_visual_tokens(cfg)
        if cfg.model.pooling != "avg":
            raise ValueError("EXP-12 requires parameter-free spatial average pooling")
        if cfg.model.duplicate_frames:
            raise ValueError("EXP-12 requires duplicate_frames=false")
        if raw_frames != units * cfg.train.temporal_patch_size:
            raise ValueError(
                "EXP-12 raw_num_frames must equal num_temporal_units * temporal_patch_size"
            )
        if cfg.train.clip_duration_seconds > 0:
            expected_duration = raw_frames / cfg.train.sample_fps
            if abs(expected_duration - cfg.train.clip_duration_seconds) > 1e-6:
                raise ValueError("raw_num_frames/sample_fps must equal clip_duration_seconds")
        if cfg.model.tokens_per_frame not in (k, 4):
            raise ValueError("legacy tokens_per_frame must either match K or remain at its default 4")
        if cfg.model.mask_variant != "v1" or cfg.model.reg_enabled or cfg.model.mtp_enabled:
            raise ValueError("EXP-12 requires clean input with legacy reg/MTP disabled")
        if cfg.model.dual_view != "off":
            raise ValueError("EXP-12 does not use dual-view or masked CE")
        if cfg.train.phase != "b":
            raise ValueError("EXP-12 is a Phase-B CE experiment")
        if cfg.train.train_vision:
            raise ValueError("EXP-12 requires one frozen visual tower and merger")
        if resolved_query_count(cfg) != k:
            raise ValueError("state_query_count must resolve to visual_tokens_per_unit")
        if cfg.train.state_horizon_units < 1 or cfg.train.state_horizon_units >= units:
            raise ValueError("state_horizon_units must be in [1, num_temporal_units)")
        expected_horizon = (
            cfg.train.state_horizon_units * cfg.train.temporal_patch_size / cfg.train.sample_fps
        )
        if abs(expected_horizon - cfg.train.state_horizon_seconds) > 1e-6:
            raise ValueError("state_horizon_seconds does not match units, patch size, and fps")
        if cfg.model.event_condition_enable:
            if cfg.model.state_predictor_mode != "observation_event_query":
                raise ValueError("event_condition_enable requires observation_event_query mode")
            if cfg.model.num_horizon_embeddings <= len(cfg.model.state_horizon_values_seconds):
                raise ValueError("event mode needs one reserved event-horizon embedding")
            if resolved_query_count(cfg, event=True) != k:
                raise ValueError("event_query_count must resolve to visual_tokens_per_unit")
        elif cfg.model.state_predictor_mode == "observation_event_query":
            raise ValueError("observation_event_query requires event_condition_enable=true")
    if (
        not cfg.model.reg_enabled
        and not cfg.model.mtp_enabled
        and not cfg.model.orca_enabled
        and cfg.model.dual_view != "ce"
        and cfg.train.lambda_reg != 0
    ):
        raise ValueError("nonzero lambda_reg requires a latent objective")
    return cfg
