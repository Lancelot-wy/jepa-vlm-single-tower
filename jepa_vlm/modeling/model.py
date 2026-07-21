"""JEPA-style masked latent regression on the Qwen3-VL LLM backbone.

Data flow (Phase A):
  frames -> Qwen3VLVisionModel -> merged tokens per latent frame (grid Hm x Wm)
         -> per-frame pooling to P=4 tokens  => h  (B, T, P, D)   [LLM input AND regression target]
  target = LayerNorm(h).detach()            (stop-grad, no EMA)
  mask   : whole latent frames replaced by learnable [M] embedding (tube masking)
  DeepStack features are pooled the same way and ZEROED at masked positions
  (otherwise the target leaks into early decoder layers - Qwen3-VL specific pitfall).
  LLM (causal, native MRoPE with per-frame 1x2x2 grids) -> hidden states
  reg head : hidden[t] -> h_t          (V1: all positions / V2.1: [M] only / V2.2: all)
  MTP heads: hidden[t] -> h_{t+1..t+k} (non-masked source positions, token-aligned)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    Config,
    is_exp12_config,
    resolved_query_count,
    resolved_visual_tokens,
)
from ..data.state_sampler import sample_state_pairs
from .heads import MLPHead, MTPHeads
from .masking import sample_token_mask
from .pooling import AttnPool, SpatialVisualTokenPooler, avg_pool_frames, resolve_pooled_grid
from .state_loss import DistributedRunningCenter, compute_state_objective
from .state_prediction import (
    StateQueryBuilder,
    TransitionHead,
    horizon_embedding_id,
    query_position_ids,
)

POOL_SIDE = 2  # pooled grid side -> tokens_per_frame = POOL_SIDE**2


@dataclass
class JepaOutput:
    loss: torch.Tensor | None = None
    reg_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    ce_loss: torch.Tensor | None = None
    ce_per_sample: torch.Tensor | None = None  # (B,) answer-token CE; used for likelihood scoring
    metrics: dict | None = None
    hidden_states: tuple | None = None
    token_mask: torch.Tensor | None = None
    target: torch.Tensor | None = None
    state_loss: torch.Tensor | None = None
    event_loss: torch.Tensor | None = None


def visual_only_position_ids(
    B: int,
    T: int,
    device,
    pooled_grid: tuple[int, int] = (POOL_SIDE, POOL_SIDE),
) -> torch.Tensor:
    """MRoPE IDs for T temporal units and one row-major pooled spatial grid.

    Returns (4, B, L): row 0 = text positions (arange, used for causal-mask bookkeeping),
    rows 1..3 = (t, h, w).
    """
    grid_h, grid_w = pooled_grid
    P = grid_h * grid_w
    L = T * P
    stride = max(grid_h, grid_w)
    starts = torch.arange(T, device=device) * stride
    t_ids = starts.repeat_interleave(P)
    row_offsets = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    col_offsets = torch.arange(grid_w, device=device).repeat(grid_h)
    h_ids = (starts[:, None] + row_offsets[None, :]).reshape(-1)
    w_ids = (starts[:, None] + col_offsets[None, :]).reshape(-1)
    text_row = torch.arange(L, device=device)
    pos = torch.stack([text_row, t_ids, h_ids, w_ids], dim=0)    # (4, L)
    return pos[:, None, :].expand(-1, B, -1).contiguous()


def mixed_position_ids(
    input_ids: torch.Tensor,
    video_mask: torch.Tensor,
    num_frames: int,
    pooled_grid: tuple[int, int] = (POOL_SIDE, POOL_SIDE),
) -> torch.Tensor:
    """MRoPE ids for text + video sequences (Phase B).

    Each sample must contain exactly ``num_frames * pooled_grid_area`` video
    placeholders, contiguous per temporal unit. Returns ``[4,B,L]``.
    """
    B, L = input_ids.shape
    grid_h, grid_w = pooled_grid
    P = grid_h * grid_w
    device = input_ids.device
    pos = torch.zeros(4, B, L, dtype=torch.long, device=device)
    row_offsets = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    col_offsets = torch.arange(grid_w, device=device).repeat(grid_h)
    frame_offsets = torch.stack(
        [torch.zeros(P, dtype=torch.long, device=device), row_offsets, col_offsets], dim=0
    )
    stride = max(grid_h, grid_w)
    for b in range(B):
        vm = video_mask[b]
        cur = 0
        i = 0
        while i < L:
            if not vm[i]:
                j = i
                while j < L and not vm[j]:
                    j += 1
                n = j - i
                block = torch.arange(n, device=device) + cur
                pos[0, b, i:j] = block
                pos[1:, b, i:j] = block
                cur += n
                i = j
            else:
                # one latent frame = P contiguous video tokens
                if i + P > L or not bool(vm[i:i + P].all()):
                    raise ValueError("video placeholders are not aligned to pooled temporal units")
                pos[0, b, i:i + P] = torch.arange(P, device=device) + cur
                pos[1:, b, i:i + P] = frame_offsets + cur
                cur += stride
                i += P
    return pos


class JepaQwen3VL(nn.Module):
    def __init__(self, hf_model, cfg: Config):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        self.visual = hf_model.model.visual
        self.language_model = hf_model.model.language_model
        self.lm_head = hf_model.lm_head  # Phase B only; frozen & unused in Phase A
        self.hf_config = hf_model.config

        D = self.hf_config.text_config.hidden_size
        self.hidden_size = D
        self.exp12_enabled = is_exp12_config(cfg)
        self.visual_tokens_per_unit = resolved_visual_tokens(cfg)
        self.mask_embed = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.mask_embed, std=0.02)

        self.pooling = mc.pooling
        if self.exp12_enabled:
            self.visual_pooler = SpatialVisualTokenPooler(self.visual_tokens_per_unit)
        else:
            self.visual_pooler = None
        if mc.pooling == "attn":
            self.attn_pool = AttnPool(D, num_queries=mc.tokens_per_frame)
        if not self.exp12_enabled:
            assert mc.tokens_per_frame == POOL_SIDE * POOL_SIDE, "legacy path supports 2x2 pooling"

        self.reg_head = MLPHead(D, mc.reg_head_hidden)
        self.mtp_heads = MTPHeads(D, mc.mtp_k, mc.reg_head_hidden) if mc.mtp_enabled else None
        if mc.orca_enabled:
            if mc.orca_use_queries:
                self.orca_queries = nn.Parameter(torch.empty(mc.orca_query_tokens, D))
                nn.init.normal_(self.orca_queries, std=0.02)
            else:
                self.register_parameter("orca_queries", None)
            self.orca_head = MLPHead(D, mc.reg_head_hidden)
        else:
            self.register_parameter("orca_queries", None)
            self.orca_head = None
        self.target_norm = nn.LayerNorm(D, elementwise_affine=False)

        # EXP-12 state modules.  Pure CE arms intentionally create no query or
        # transition-head parameters, so the optimizer audit can prove the only
        # A0/A2/A4 change is K.
        self.state_query_builder = None
        self.event_query_builder = None
        self.state_transition_head = None
        self.event_direction_embedding = None
        self.state_center = None
        if mc.state_predictor_mode != "none":
            k = self.visual_tokens_per_unit
            self.state_transition_head = TransitionHead(D, mc.state_head_hidden)
            if mc.state_predictor_mode in ("query", "observation_event_query"):
                self.state_query_builder = StateQueryBuilder(
                    D, resolved_query_count(cfg), mc.num_horizon_embeddings,
                    mc.state_query_position_encoding,
                )
            if mc.state_predictor_mode == "observation_event_query":
                self.event_query_builder = StateQueryBuilder(
                    D, resolved_query_count(cfg, event=True), mc.num_horizon_embeddings,
                    mc.state_query_position_encoding,
                )
                self.event_direction_embedding = nn.Embedding(2, D)
                nn.init.normal_(self.event_direction_embedding.weight, std=0.02)
            self.state_center = DistributedRunningCenter(D, mc.state_center_momentum)
            if resolved_query_count(cfg) != k:
                raise ValueError("state query count must equal visual token count")

        if mc.bidirectional_visual and mc.mtp_enabled:
            raise ValueError("bidirectional_visual sees future frames -> MTP is trivial; "
                             "set model.mtp_enabled=false for this ablation")
        if mc.residual_target and (
            mc.mask_variant != "v2.1" or mc.mask_mode != "tube" or mc.mtp_enabled
        ):
            raise ValueError("residual_target requires mask_variant=v2.1, mask_mode=tube "
                             "and mtp_enabled=false (residual semantics are defined per "
                             "masked frame against its nearest visible frame)")

    # ------------------------------------------------------------------ vision
    def encode_video(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """pixel_values: (B, S, patch_dim) patchified videos (identical grids across batch).
        grid_thw: (3,) or (B, 3) in patch units.

        Returns pooled tokens, pooled DeepStack features, and pooled HxW."""
        B = pixel_values.shape[0]
        grid_thw = grid_thw.to(pixel_values.device)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw[None, :].expand(B, -1)
        T, Hp, Wp = (int(x) for x in grid_thw[0])
        Hm, Wm = Hp // 2, Wp // 2
        flat = pixel_values.reshape(-1, pixel_values.shape[-1])
        out = self.visual(flat, grid_thw)
        merged = out.pooler_output.reshape(B, T, Hm, Wm, -1)
        if self.exp12_enabled:
            h, pooled_grid = self.visual_pooler(merged)
        elif self.pooling == "attn":
            h = self.attn_pool(merged)
            pooled_grid = resolve_pooled_grid(h.shape[-2], Hm, Wm)
        else:
            h = avg_pool_frames(merged, POOL_SIDE)
            pooled_grid = (POOL_SIDE, POOL_SIDE)
        deepstack = []
        if self.cfg.model.use_deepstack:
            for feat in out.deepstack_features:
                reshaped = feat.reshape(B, T, Hm, Wm, -1)
                if self.exp12_enabled:
                    pooled, level_grid = self.visual_pooler(reshaped, pooled_grid)
                    if level_grid != pooled_grid:
                        raise AssertionError("DeepStack used a different pooled grid")
                else:
                    pooled = avg_pool_frames(reshaped, POOL_SIDE)
                if pooled.shape[-2] != h.shape[-2]:
                    raise AssertionError("DeepStack token count differs from main visual tokens")
                deepstack.append(pooled)
        if h.shape[-2] != self.visual_tokens_per_unit:
            raise AssertionError("main visual token count does not match configured K")
        return h, deepstack, pooled_grid

    def encode_temporal_units_independently(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """Encode each temporal patch as an independent visual input.

        The normal VQA path intentionally keeps Qwen's native full-video encoder.
        This teacher path is used only by the Orca-inspired transition objective:
        current and target states cannot exchange information inside the ViT, and
        the frozen visual encoder provides a stationary regression target.
        """
        B = pixel_values.shape[0]
        grid_thw = grid_thw.to(pixel_values.device)
        if grid_thw.ndim > 1:
            first = grid_thw[0]
            if not torch.equal(grid_thw, first[None].expand_as(grid_thw)):
                raise ValueError("independent-frame encoding requires identical grids in a batch")
            grid_thw = first
        T, Hp, Wp = (int(x) for x in grid_thw)
        patches_per_frame = Hp * Wp
        expected = T * patches_per_frame
        if pixel_values.shape[1] != expected:
            raise ValueError(
                f"pixel/grid mismatch: got {pixel_values.shape[1]} patches, expected {expected}"
            )
        per_frame = pixel_values.reshape(B * T, patches_per_frame, pixel_values.shape[-1])
        one_frame_grids = torch.tensor(
            [1, Hp, Wp], dtype=grid_thw.dtype, device=grid_thw.device
        )[None].expand(B * T, -1)
        out = self.visual(per_frame.reshape(-1, per_frame.shape[-1]), one_frame_grids)
        Hm, Wm = Hp // 2, Wp // 2
        merged = out.pooler_output.reshape(B, T, Hm, Wm, -1)
        if self.exp12_enabled:
            return self.visual_pooler(merged)
        if self.pooling == "attn":
            result = self.attn_pool(merged)
            return result, resolve_pooled_grid(result.shape[-2], Hm, Wm)
        return avg_pool_frames(merged, POOL_SIDE), (POOL_SIDE, POOL_SIDE)

    # Historical EXP-11 API name retained for old configs/checkpoints.
    def encode_frames_independently(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        states, _ = self.encode_temporal_units_independently(pixel_values, grid_thw)
        return states

    def _encode_video_for_llm(self, pixel_values, grid_thw):
        """Run the single physical visual module, avoiding visual autograd when frozen."""
        frozen = not any(parameter.requires_grad for parameter in self.visual.parameters())
        if frozen:
            with torch.inference_mode():
                h, deepstack, pooled_grid = self.encode_video(pixel_values, grid_thw)
            # Inference tensors cannot be saved by trainable LLM layers for
            # backward, so materialize ordinary detached tensors afterwards.
            h = h.detach().clone()
            deepstack = [feature.detach().clone() for feature in deepstack]
            return h, deepstack, pooled_grid
        return self.encode_video(pixel_values, grid_thw)

    def _encode_frozen_temporal_units(self, pixel_values, grid_thw):
        if any(parameter.requires_grad for parameter in self.visual.parameters()):
            raise AssertionError("EXP-12 state target requires a frozen visual module")
        with torch.inference_mode():
            states, pooled_grid = self.encode_temporal_units_independently(pixel_values, grid_thw)
        states = states.detach().clone()
        if states.requires_grad:
            raise AssertionError("frozen visual states unexpectedly require gradients")
        return states, pooled_grid

    # ------------------------------------------------------------------ mask & embed
    def apply_mask(self, h: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Replace masked tokens with the learnable [M] embedding. h: (B,T,P,D)."""
        m = token_mask.to(h.device).unsqueeze(-1)
        return torch.where(m, self.mask_embed.to(h.dtype).expand_as(h), h)

    def _deepstack_embeds(self, deepstack: list[torch.Tensor], token_mask: torch.Tensor | None):
        """Zero deepstack features at masked positions, flatten to (B*T*P, D) per level."""
        if not deepstack:
            return None
        outs = []
        for feat in deepstack:
            if feat.shape[-2] != self.visual_tokens_per_unit:
                raise AssertionError("DeepStack feature does not have configured K tokens")
            if token_mask is not None:
                feat = feat.masked_fill(token_mask.to(feat.device).unsqueeze(-1), 0.0)
            outs.append(feat.reshape(-1, feat.shape[-1]))
        return outs

    def _bidirectional_mask(self, B: int, L: int, dtype, device) -> torch.Tensor:
        """4D float mask letting all visual tokens attend to each other (ablation)."""
        allowed = torch.ones(L, L, dtype=torch.bool, device=device)
        neg = torch.finfo(dtype).min
        mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=device), neg)
        return mask[None, None].expand(B, 1, L, L)

    # ------------------------------------------------------------------ forward (Phase A)
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        disable_mask: bool = False,
        generator: torch.Generator | None = None,
        state_eligible: torch.Tensor | None = None,
        event_batch: dict[str, torch.Tensor] | None = None,
    ) -> JepaOutput:
        mc = self.cfg.model
        h, deepstack, pooled_grid = self._encode_video_for_llm(pixel_values, grid_thw)
        B, T, P, D = h.shape

        if self.exp12_enabled:
            return self._forward_exp12(
                h, deepstack, pooled_grid, pixel_values, grid_thw,
                input_ids, attention_mask, labels, output_hidden_states,
                state_eligible=state_eligible, event_batch=event_batch,
                disable_state=disable_mask,
            )

        # normed_online keeps the graph to the encoder (variance regularizer);
        # the regression target is its detached copy (stop-grad, no EMA).
        normed_online = self.target_norm(h.float())
        target = normed_online.detach()

        if mc.orca_enabled and not disable_mask:
            if input_ids is None or labels is None:
                raise ValueError("orca_enabled is a Phase-B joint CE + transition objective")
            return self._forward_orca_joint(
                h, deepstack, pixel_values, grid_thw,
                input_ids, attention_mask, labels, output_hidden_states,
            )

        use_mask = (mc.mask_variant != "v1") and not disable_mask
        if use_mask and token_mask is None:
            token_mask, _ = sample_token_mask(
                B, T, P, mc.mask_ratio, mode=mc.mask_mode,
                max_run=mc.mask_tube_max_run, generator=generator,
            )
        if token_mask is not None:
            token_mask = token_mask.to(h.device)
        h_in = self.apply_mask(h, token_mask) if use_mask else h

        if input_ids is None:
            return self._forward_visual_only(
                h_in, deepstack, target, token_mask if use_mask else None,
                output_hidden_states, normed_online,
            )
        if use_mask and mc.dual_view != "off" and labels is not None:
            return self._forward_dual_view(
                h, h_in, deepstack, target, token_mask,
                input_ids, attention_mask, labels, output_hidden_states, normed_online,
            )
        return self._forward_with_text(
            h_in, deepstack, target, token_mask if use_mask else None,
            input_ids, attention_mask, labels, output_hidden_states, normed_online,
        )

    def _forward_exp12(
        self,
        h,
        deepstack,
        pooled_grid,
        pixel_values,
        grid_thw,
        input_ids,
        attention_mask,
        labels,
        output_hidden_states,
        *,
        state_eligible,
        event_batch,
        disable_state,
    ):
        """Clean answer CE plus an isolated optional state-prediction forward."""
        if input_ids is None or labels is None:
            raise ValueError("EXP-12 requires Phase-B input_ids and answer-only labels")
        B, T, K, D = h.shape
        if K != self.visual_tokens_per_unit:
            raise AssertionError("pooled visual shape does not match configured K")
        for feature in deepstack:
            if feature.shape[-2] != K:
                raise AssertionError("DeepStack feature was not pooled to K")
        target = self.target_norm(h.float()).detach()
        out_ce = self._forward_with_text(
            h, deepstack, target, None, input_ids, attention_mask, labels,
            output_hidden_states, normed_online=None, compute_reg=False,
            pooled_grid=pooled_grid,
        )
        metrics = dict(out_ce.metrics or {})
        metrics.update({
            "model/visual_tokens_per_unit": float(K),
            "model/total_video_visual_tokens": float(T * K),
            "model/state_query_count": float(
                resolved_query_count(self.cfg)
                if self.cfg.model.state_predictor_mode != "none" else 0
            ),
            "model/pooled_grid_h": float(pooled_grid[0]),
            "model/pooled_grid_w": float(pooled_grid[1]),
            "model/llm_sequence_length": float(input_ids.shape[1]),
            "model/deepstack_token_count": float(K if deepstack else 0),
        })
        total_loss = out_ce.ce_loss
        state_loss = None
        event_loss = None
        state_target = None
        mode = self.cfg.model.state_predictor_mode
        if mode != "none" and not disable_state:
            states, state_grid = self._encode_frozen_temporal_units(pixel_values, grid_thw)
            if state_grid != pooled_grid:
                raise AssertionError("CE and state branches resolved different pooled grids")
            if states.shape != h.shape:
                raise AssertionError("independent state and CE visual shapes differ")
            target_states = self.target_norm(states.float()).detach()
            if target_states.requires_grad:
                raise AssertionError("EXP-12 target unexpectedly requires gradients")
            pairs = sample_state_pairs(
                states, target_states, self.cfg.train.state_horizon_units, state_eligible
            )
            state_loss, state_metrics = self._run_state_transition(pairs, pooled_grid, mode)
            metrics.update(state_metrics)
            metrics["state/frame_encoding_mse"] = float(
                F.mse_loss(h.detach().float(), states.float()).detach()
            )
            metrics["state/eligible_fraction"] = float(pairs.valid.float().mean().detach())
            total_loss = total_loss + self.cfg.model.state_loss_weight * state_loss
            metrics["state/weighted_loss"] = float(
                (self.cfg.model.state_loss_weight * state_loss).detach()
            )
            state_target = pairs.target

            if self.cfg.model.event_condition_enable and event_batch is not None:
                event_loss, event_metrics = self._run_event_transition(event_batch)
                total_loss = total_loss + self.cfg.model.event_loss_weight * event_loss
                metrics.update(event_metrics)
                metrics["event/weighted_loss"] = float(
                    (self.cfg.model.event_loss_weight * event_loss).detach()
                )

        return JepaOutput(
            loss=total_loss,
            ce_loss=out_ce.ce_loss,
            ce_per_sample=out_ce.ce_per_sample,
            metrics=metrics,
            hidden_states=out_ce.hidden_states,
            target=state_target,
            state_loss=state_loss,
            event_loss=event_loss,
        )

    def _run_state_transition(self, pairs, pooled_grid, mode: str):
        mc, tc = self.cfg.model, self.cfg.train
        source = pairs.source
        n_pairs, K, D = source.shape
        source_pos = visual_only_position_ids(n_pairs, 1, source.device, pooled_grid)
        if mode in ("query", "observation_event_query"):
            horizon_id = horizon_embedding_id(
                tc.state_horizon_seconds, mc.state_horizon_values_seconds
            )
            queries = self.state_query_builder(
                n_pairs, pooled_grid, horizon_id, dtype=source.dtype, device=source.device
            )
            instruction_id = getattr(self.hf_config, "vision_start_token_id")
            instruction_ids = torch.full(
                (n_pairs, 1), instruction_id, dtype=torch.long, device=source.device
            )
            instruction = self.language_model.embed_tokens(instruction_ids).to(source.dtype)
            inputs = torch.cat([source, instruction, queries], dim=1)
            stride = max(pooled_grid)
            instruction_pos = torch.full(
                (4, n_pairs, 1), stride, dtype=torch.long, device=source.device
            )
            query_pos = query_position_ids(
                n_pairs, pooled_grid, stride + 1, source.device
            )
            position_ids = torch.cat([source_pos, instruction_pos, query_pos], dim=-1)
            pred_slice = slice(K + 1, K + 1 + K)
        elif mode == "no_query":
            inputs = source.to(self.state_transition_head.net[0].weight.dtype)
            position_ids = source_pos
            pred_slice = slice(0, K)
        else:
            raise ValueError(f"unsupported state transition mode {mode}")
        outputs = self.language_model(
            inputs_embeds=inputs,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )
        pred_hidden = outputs.last_hidden_state[:, pred_slice]
        pred = self.state_transition_head(pred_hidden).float()
        if pred.shape[-2] != K or pairs.target.shape[-2] != K:
            raise AssertionError("query, prediction, and target must all contain K tokens")
        result = compute_state_objective(
            pred, pairs.target, pairs.source_target_space, pairs.valid, self.state_center,
            dynamic_threshold=mc.state_dynamic_threshold,
            dynamic_weighting=mc.state_dynamic_weighting,
            beat_copy_loss_weight=mc.beat_copy_loss_weight,
            beat_copy_margin=mc.beat_copy_margin,
            update_center=self.training,
            prefix="state",
        )
        result.metrics["state/horizon_units"] = float(tc.state_horizon_units)
        result.metrics["state/horizon_seconds"] = float(tc.state_horizon_seconds)
        result.metrics["state/query_enabled"] = float(mode != "no_query")
        return result.loss, result.metrics

    @staticmethod
    def _select_fractional_unit(states: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
        B, T, K, D = states.shape
        ids = (fractions.to(states.device).clamp(0, 1) * (T - 1)).round().long()
        return states[torch.arange(B, device=states.device), ids]

    def _run_event_transition(self, batch: dict[str, torch.Tensor]):
        """Observation + condition + Event Query; target never enters the LLM."""
        mc = self.cfg.model
        source_states, source_grid = self._encode_frozen_temporal_units(
            batch["source_pixel_values"], batch["source_grid_thw"]
        )
        target_states, target_grid = self._encode_frozen_temporal_units(
            batch["target_pixel_values"], batch["target_grid_thw"]
        )
        negative_states, negative_grid = self._encode_frozen_temporal_units(
            batch["negative_pixel_values"], batch["negative_grid_thw"]
        )
        if not (source_grid == target_grid == negative_grid):
            raise AssertionError("event source/target/negative grids differ")
        source = self._select_fractional_unit(source_states, batch["source_inner_fraction"])
        target = self.target_norm(
            self._select_fractional_unit(target_states, batch["target_inner_fraction"]).float()
        ).detach()
        negative = self.target_norm(
            self._select_fractional_unit(negative_states, batch["target_inner_fraction"]).float()
        ).detach()
        current = self.target_norm(source.float()).detach()
        if target.requires_grad or negative.requires_grad:
            raise AssertionError("event targets must be detached")
        B, K, D = source.shape
        condition_ids = batch["condition_input_ids"]
        condition_mask = batch["condition_attention_mask"].bool()
        condition = self.language_model.embed_tokens(condition_ids).to(source.dtype)
        direction = self.event_direction_embedding(batch["direction"]).unsqueeze(1).to(source.dtype)
        event_horizon_id = len(mc.state_horizon_values_seconds)
        queries = self.event_query_builder(
            B, source_grid, event_horizon_id, dtype=source.dtype, device=source.device
        )
        inputs = torch.cat([source, direction, condition, queries], dim=1)
        source_pos = visual_only_position_ids(B, 1, source.device, source_grid)
        rest = inputs.shape[1] - K
        rest_pos = torch.arange(K, K + rest, device=source.device)
        rest_pos = rest_pos[None, None].expand(4, B, -1)
        position_ids = torch.cat([source_pos, rest_pos], dim=-1)
        attention_mask = torch.cat([
            torch.ones(B, K + 1, dtype=torch.long, device=source.device),
            condition_mask.long(),
            torch.ones(B, K, dtype=torch.long, device=source.device),
        ], dim=1)
        outputs = self.language_model(
            inputs_embeds=inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        pred_hidden = outputs.last_hidden_state[:, -K:]
        pred = self.state_transition_head(pred_hidden).float()
        valid = torch.ones(B, K, dtype=torch.bool, device=source.device)
        result = compute_state_objective(
            pred, target, current, valid, self.state_center,
            dynamic_threshold=mc.event_dynamic_threshold,
            dynamic_weighting=mc.state_dynamic_weighting,
            beat_copy_loss_weight=mc.beat_copy_loss_weight,
            beat_copy_margin=mc.beat_copy_margin,
            update_center=self.training,
            prefix="event",
            negative_target=negative,
        )
        direction_values = batch["direction"].float()
        result.metrics["event/forward_fraction"] = float(direction_values.mean().detach())
        result.metrics["event/backward_fraction"] = 1.0 - result.metrics["event/forward_fraction"]
        return result.loss, result.metrics

    def _forward_orca_joint(self, h, deepstack, pixel_values, grid_thw,
                            input_ids, attention_mask, labels, output_hidden_states):
        """Clean VQA/CE plus a short observation-only predictive-query branch."""
        out_ce = self._forward_with_text(
            h, deepstack, self.target_norm(h.float()).detach(), None,
            input_ids, attention_mask, labels, output_hidden_states,
            normed_online=None, compute_reg=False,
        )
        # The pilot explicitly freezes the ViT.  no_grad also prevents retaining
        # an unnecessary second visual graph for the independently encoded states.
        with torch.no_grad():
            states = self.encode_frames_independently(pixel_values, grid_thw)
            target_states = self.target_norm(states.float())
        transition_loss, transition_metrics = self._orca_transition(states, target_states)
        # Qwen3-VL already builds per-temporal-unit cu_seqlens, so the native
        # full-video ViT path should be equivalent to explicitly batching frames
        # as separate one-frame entries.  Log the invariant instead of assuming
        # leakage (or assuming its absence) from code inspection alone.
        native_vs_independent = F.mse_loss(h.detach().float(), states.float())
        transition_metrics["orca_frame_encoding_mse"] = float(native_vs_independent)
        lam = self.cfg.train.lambda_reg
        loss = out_ce.ce_loss + lam * transition_loss
        metrics = {**(out_ce.metrics or {}), **transition_metrics}
        metrics["orca_weighted_loss"] = float((lam * transition_loss).detach())
        return JepaOutput(
            loss=loss, reg_loss=transition_loss, ce_loss=out_ce.ce_loss,
            ce_per_sample=out_ce.ce_per_sample, metrics=metrics,
            hidden_states=out_ce.hidden_states, target=target_states.detach(),
        )

    def _orca_transition(self, states: torch.Tensor, target_states: torch.Tensor):
        """Predict a future frozen visual state from one frame plus query tokens.

        Each frame pair is an independent short causal sequence
        ``[4 current-state tokens, 4 learned query tokens]``.  Query hidden states
        are projected to the four target-state tokens by a two-layer MLP.
        """
        mc = self.cfg.model
        B, T, P, D = states.shape
        gap = mc.orca_target_gap
        n_pairs = B * (T - gap)
        source = states[:, : T - gap].reshape(n_pairs, P, D)
        source_target = target_states[:, : T - gap].reshape(n_pairs, P, D)
        target = target_states[:, gap:].reshape(n_pairs, P, D).detach()
        visual_pos = visual_only_position_ids(n_pairs, 1, states.device)
        if mc.orca_use_queries:
            source = source.to(self.orca_queries.dtype)
            queries = self.orca_queries[None].expand(n_pairs, -1, -1)
            inputs = torch.cat([source, queries], dim=1)
            query_pos = torch.arange(P, P + mc.orca_query_tokens, device=states.device)
            query_pos = query_pos[None, None].expand(4, n_pairs, -1)
            position_ids = torch.cat([visual_pos, query_pos], dim=-1)
        else:
            inputs = source.to(self.orca_head.net[0].weight.dtype)
            position_ids = visual_pos
        outputs = self.language_model(
            inputs_embeds=inputs,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )
        pred_hidden = outputs.last_hidden_state[:, P:] if mc.orca_use_queries else outputs.last_hidden_state
        pred = self.orca_head(pred_hidden).float()
        loss = F.mse_loss(pred, target)
        persistence = F.mse_loss(source_target, target)
        ratio = float(loss.detach() / persistence.clamp_min(1e-8))
        metrics = {
            "orca_loss": float(loss.detach()),
            "orca_persistence_mse": float(persistence.detach()),
            "orca_persistence_ratio": ratio,
            "orca_gain_vs_persistence": 1.0 - ratio,
            "orca_target_std": float(target.std(dim=(0, 1)).mean()),
            "orca_pred_std": float(pred.detach().std(dim=(0, 1)).mean()),
            "orca_target_gap": float(gap),
            "orca_use_queries": float(mc.orca_use_queries),
        }
        return loss, metrics

    def _forward_dual_view(self, h_clean, h_masked, deepstack, target, token_mask,
                           input_ids, attention_mask, labels, output_hidden_states,
                           normed_online):
        """V4 dual-view: the CE view never sees [M] (train == deploy distribution);
        the masked view carries the latent-prediction pressure.

        dual_view=reg: L = CE(clean) + lambda_reg * (reg+mtp)(masked, visual-only pass)
        dual_view=ce : L = 0.5*(CE(clean) + CE(masked)); no regression (R-Drop-style
                       consistency control arm - isolates regularization from prediction).
        """
        mc = self.cfg.model
        out_ce = self._forward_with_text(
            h_clean, deepstack, target, None,
            input_ids, attention_mask, labels, output_hidden_states, normed_online,
            compute_reg=False,
        )
        if mc.dual_view == "ce":
            out_m = self._forward_with_text(
                h_masked, deepstack, target, token_mask,
                input_ids, attention_mask, labels, False, normed_online,
                compute_reg=False,
            )
            loss = 0.5 * (out_ce.ce_loss + out_m.ce_loss)
            metrics = dict(out_ce.metrics or {})
            metrics["ce_loss_masked"] = float(out_m.ce_loss.detach())
            return JepaOutput(
                loss=loss, ce_loss=out_ce.ce_loss, ce_per_sample=out_ce.ce_per_sample,
                metrics=metrics, hidden_states=out_ce.hidden_states,
                token_mask=token_mask, target=target,
            )
        out_reg = self._forward_visual_only(
            h_masked, deepstack, target, token_mask, False, normed_online,
        )
        lam = self.cfg.train.lambda_reg
        loss = out_ce.ce_loss + lam * out_reg.loss if out_reg.loss is not None else out_ce.ce_loss
        metrics = {**(out_reg.metrics or {}), "ce_loss": float(out_ce.ce_loss.detach())}
        return JepaOutput(
            loss=loss, reg_loss=out_reg.reg_loss, mtp_loss=out_reg.mtp_loss,
            ce_loss=out_ce.ce_loss, ce_per_sample=out_ce.ce_per_sample, metrics=metrics,
            hidden_states=out_ce.hidden_states, token_mask=token_mask, target=target,
        )

    def _forward_visual_only(self, h_in, deepstack, target, token_mask,
                             output_hidden_states, normed_online=None,
                             pooled_grid: tuple[int, int] | None = None):
        mc = self.cfg.model
        B, T, P, D = h_in.shape
        L = T * P
        device = h_in.device
        inputs_embeds = h_in.reshape(B, L, D)
        pooled_grid = pooled_grid or (POOL_SIDE, POOL_SIDE)
        if pooled_grid[0] * pooled_grid[1] != P:
            raise ValueError("pooled grid does not match visual token count")
        position_ids = visual_only_position_ids(B, T, device, pooled_grid)
        visual_pos_masks = torch.ones(B, L, dtype=torch.bool, device=device)
        ds_embeds = self._deepstack_embeds(deepstack, token_mask)

        attn_mask = None
        if mc.bidirectional_visual:
            attn_mask = self._bidirectional_mask(B, L, inputs_embeds.dtype, device)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attn_mask,
            visual_pos_masks=visual_pos_masks if ds_embeds is not None else None,
            deepstack_visual_embeds=ds_embeds,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        hidden = outputs.last_hidden_state.reshape(B, T, P, D)
        loss, reg_loss, mtp_loss, metrics = self._compute_losses(
            hidden, target, token_mask, normed_online)
        return JepaOutput(
            loss=loss, reg_loss=reg_loss, mtp_loss=mtp_loss, metrics=metrics,
            hidden_states=getattr(outputs, "hidden_states", None),
            token_mask=token_mask, target=target,
        )

    def _forward_with_text(self, h_in, deepstack, target, token_mask,
                           input_ids, attention_mask, labels, output_hidden_states,
                           normed_online=None, compute_reg=True,
                           pooled_grid: tuple[int, int] | None = None):
        B, T, P, D = h_in.shape
        video_token_id = self.hf_config.video_token_id
        video_mask = input_ids == video_token_id  # (B, L)
        assert int(video_mask.sum()) == B * T * P, "each sample needs exactly T*P video tokens"

        pooled_grid = pooled_grid or (POOL_SIDE, POOL_SIDE)
        if pooled_grid[0] * pooled_grid[1] != P:
            raise ValueError("pooled grid does not match visual token count")

        inputs_embeds = self.language_model.embed_tokens(input_ids).clone()
        inputs_embeds[video_mask] = h_in.reshape(-1, D).to(inputs_embeds.dtype)
        position_ids = mixed_position_ids(input_ids, video_mask, T, pooled_grid)
        ds_embeds = self._deepstack_embeds(deepstack, token_mask)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=video_mask if ds_embeds is not None else None,
            deepstack_visual_embeds=ds_embeds,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        seq_hidden = outputs.last_hidden_state
        hidden = seq_hidden[video_mask].reshape(B, T, P, D)
        if compute_reg:
            loss, reg_loss, mtp_loss, metrics = self._compute_losses(
                hidden, target, token_mask, normed_online)
        else:
            loss, reg_loss, mtp_loss, metrics = None, None, None, {}

        ce_loss = None
        if labels is not None:
            logits = self.lm_head(seq_hidden)
            # unreduced CE, then the same global token-mean as before plus a per-sample
            # mean (likelihood scoring for QA eval, e.g. temporal_qa_eval)
            ce_tok = F.cross_entropy(
                logits[:, :-1].float().transpose(1, 2),
                labels[:, 1:],
                ignore_index=-100,
                reduction="none",
            )  # (B, L-1), zeros at ignored positions
            valid = labels[:, 1:] != -100
            ce_loss = ce_tok.sum() / valid.sum().clamp_min(1)
            ce_per_sample = (ce_tok.sum(1) / valid.sum(1).clamp_min(1)).detach()
            metrics["ce_loss"] = float(ce_loss.detach())
            lam = self.cfg.train.lambda_reg
            loss = ce_loss + lam * loss if loss is not None else ce_loss
        return JepaOutput(
            loss=loss, reg_loss=reg_loss, mtp_loss=mtp_loss, ce_loss=ce_loss,
            ce_per_sample=ce_per_sample if labels is not None else None, metrics=metrics,
            hidden_states=getattr(outputs, "hidden_states", None),
            token_mask=token_mask, target=target,
        )

    # ------------------------------------------------------------------ losses & monitors
    @staticmethod
    def _nearest_visible(frame_masked_row: torch.Tensor) -> list[tuple[int, int]]:
        """(t_masked, t_src) pairs: nearest past visible frame, else nearest future one.
        Matches the causal copy baseline (and defines the residual-target reference)."""
        masked_idx = frame_masked_row.nonzero().flatten()
        free_idx = (~frame_masked_row).nonzero().flatten()
        pairs = []
        if len(masked_idx) == 0 or len(free_idx) == 0:
            return pairs
        for t in masked_idx.tolist():
            past = free_idx[free_idx < t]
            src = int(past.max()) if len(past) else int(free_idx[free_idx > t].min())
            pairs.append((t, src))
        return pairs

    def _compute_losses(self, hidden, target, token_mask, normed_online=None):
        mc = self.cfg.model
        B, T, P, D = hidden.shape
        pred = self.reg_head(hidden).float() if mc.reg_enabled else None

        if not mc.reg_enabled:
            reg_loss = None
        elif mc.residual_target and token_mask is not None:
            # target = h_t - h_src: the copy solution is exactly "predict zero", so any
            # ratio below 1 vs copy_mse now reflects genuinely modeled dynamics.
            frame_masked = token_mask.all(dim=-1)
            res_target = torch.zeros_like(target)
            for b in range(B):
                for t, s in self._nearest_visible(frame_masked[b]):
                    res_target[b, t] = target[b, t] - target[b, s]
            m = token_mask.to(hidden.device)
            reg_loss = F.mse_loss(pred[m], res_target[m])
        elif mc.mask_variant == "v2.1" and token_mask is not None:
            m = token_mask.to(hidden.device)
            reg_loss = F.mse_loss(pred[m], target[m])
        else:  # v1 / v2.2: all positions
            reg_loss = F.mse_loss(pred, target)

        mtp_loss = None
        mtp_metrics = {}
        if self.mtp_heads is not None:
            src_ok = torch.ones(B, T, P, dtype=torch.bool, device=hidden.device)
            if token_mask is not None:
                src_ok = ~token_mask.to(hidden.device)
            preds = self.mtp_heads(hidden)
            losses = []
            for j, pred_j in enumerate(preds, start=1):
                if T - j <= 0:
                    continue
                p_j = pred_j[:, : T - j].float()
                t_j = target[:, j:]
                m_j = src_ok[:, : T - j]
                if m_j.any():
                    l_j = F.mse_loss(p_j[m_j], t_j[m_j])
                    losses.append(l_j)
                    mtp_metrics[f"mtp_loss_k{j}"] = float(l_j.detach())
            if losses:
                mtp_loss = torch.stack(losses).mean()

        if reg_loss is None and mtp_loss is None:
            loss = None
        elif reg_loss is None:
            loss = mtp_loss
        elif mtp_loss is None:
            loss = reg_loss
        else:
            loss = reg_loss + mtp_loss

        var_metrics = {}
        if mc.var_reg_weight > 0 and normed_online is not None:
            # VICReg-style hinge on per-dim std across (B,T,P); gradient flows to the
            # encoder through normed_online and counters directional collapse.
            std = normed_online.reshape(-1, D).std(dim=0)
            var_loss = F.relu(mc.var_reg_gamma - std).mean()
            loss = var_loss * mc.var_reg_weight if loss is None else loss + mc.var_reg_weight * var_loss
            var_metrics["var_loss"] = float(var_loss.detach())

        monitor_metrics = self._monitors(target, token_mask)
        metrics = {
            **({"reg_loss": float(reg_loss.detach())} if reg_loss is not None else {}),
            **mtp_metrics,
            **var_metrics,
            **monitor_metrics,
        }
        if mtp_loss is not None:
            metrics["mtp_loss"] = float(mtp_loss.detach())
            persistence = monitor_metrics.get("mtp_persistence_mse")
            if persistence is not None and persistence > 0:
                ratio = float(mtp_loss.detach()) / persistence
                metrics["mtp_persistence_ratio"] = ratio
                metrics["mtp_gain_vs_persistence"] = 1.0 - ratio
        return loss, reg_loss, mtp_loss, metrics

    @torch.no_grad()
    def _monitors(self, target, token_mask):
        """Collapse & triviality monitors, computed on the (normed) target.

        - target_std: mean per-dim std across (B,T,P). -> 0 means representation collapse.
        - adj_cos: mean cosine similarity of adjacent frames' targets. -> 1 means frames
          indistinguishable (regression becomes trivial).
        - copy_mse: MSE of predicting each masked frame by copying its nearest unmasked
          frame (past-preferred under causal attention). Plan eval #3 compares the model's
          masked reg_loss against this; reg_loss must be clearly lower to be non-trivial.
        """
        B, T, P, D = target.shape
        out = {"target_std": float(target.std(dim=(0, 1, 2)).mean())}
        a, b = target[:, :-1], target[:, 1:]
        out["adj_cos"] = float(F.cosine_similarity(a.reshape(-1, D), b.reshape(-1, D), dim=-1).mean())
        # For a no-mask next-frame objective, copying the current target is the
        # relevant trivial baseline.  Unlike copy_mse below it is available in
        # EXP-10, where token_mask is deliberately None.
        if T > 1:
            out["mtp_persistence_mse"] = float(F.mse_loss(a, b))

        if token_mask is not None:
            out["mask_fraction"] = float(token_mask.float().mean())
            frame_masked = token_mask.all(dim=-1)  # (B, T)
            errs = []
            for bi in range(B):
                for t, src in self._nearest_visible(frame_masked[bi]):
                    errs.append(F.mse_loss(target[bi, src], target[bi, t]))
            if errs:
                out["copy_mse"] = float(torch.stack(errs).mean())
        return out

    # ------------------------------------------------------------------ probes
    @torch.no_grad()
    def extract_features(self, pixel_values, grid_thw, layers: list[int]) -> dict[int, torch.Tensor]:
        """No-mask forward; returns {layer_idx: (B, T, P, D)} hidden states at visual positions.
        layer_idx indexes the recorded decoder-layer outputs (negative ok)."""
        if self.exp12_enabled:
            h, deepstack, pooled_grid = self._encode_video_for_llm(pixel_values, grid_thw)
            target = self.target_norm(h.float()).detach()
            out = self._forward_visual_only(
                h, deepstack, target, None, True, normed_online=None,
                pooled_grid=pooled_grid,
            )
        else:
            out = self.forward(
                pixel_values, grid_thw, disable_mask=True, output_hidden_states=True
            )
        hs = out.hidden_states
        B = pixel_values.shape[0]
        feats = {}
        for li in layers:
            x = hs[li]
            feats[li] = x.reshape(B, -1, self.visual_tokens_per_unit, x.shape[-1])
        return feats

    def checkpoint_aux_state(self) -> dict:
        if self.state_center is None:
            return {}
        return {"state_center": self.state_center.state_dict()}

    def load_checkpoint_aux_state(self, state: dict) -> None:
        if self.state_center is None:
            return
        if "state_center" not in state:
            raise RuntimeError("EXP-12 checkpoint is missing the distributed running center")
        self.state_center.load_state_dict(state["state_center"])

    def assert_exp12_frozen_visual(self) -> None:
        if not self.exp12_enabled:
            return
        bad = [name for name, parameter in self.visual.named_parameters() if parameter.requires_grad]
        if bad:
            raise AssertionError(f"EXP-12 visual/merger parameters are trainable: {bad[:5]}")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------- builder
def tiny_qwen3vl_config():
    from transformers.models.qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        vision_config=dict(
            depth=4, hidden_size=64, intermediate_size=128, num_heads=4,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
            out_hidden_size=128, num_position_embeddings=256,
            deepstack_visual_indexes=[1, 2],
        ),
        text_config=dict(
            vocab_size=1024, hidden_size=128, intermediate_size=256,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=4096,
            rope_parameters=dict(rope_type="default", rope_theta=500000.0,
                                 mrope_section=[8, 4, 4], mrope_interleaved=True),
        ),
        image_token_id=7, video_token_id=8, vision_start_token_id=5, vision_end_token_id=6,
    )


def build_model(cfg: Config):
    from transformers import Qwen3VLForConditionalGeneration

    mc, tc = cfg.model, cfg.train
    dtype = getattr(torch, mc.dtype)
    if mc.tiny_config:
        hf = Qwen3VLForConditionalGeneration(tiny_qwen3vl_config()).to(dtype)
    else:
        hf = Qwen3VLForConditionalGeneration.from_pretrained(
            mc.pretrained, dtype=dtype, attn_implementation=mc.attn_implementation
        )
    model = JepaQwen3VL(hf, cfg)
    # JEPA heads (reg_head/mtp_heads) are created after the bf16 backbone and
    # default to fp32; align the whole wrapper to the backbone dtype so hidden
    # states and head weights match (reg/target losses still run in fp32 via .float()).
    model.to(dtype)

    # --- freezing ---
    model.lm_head.requires_grad_(False)
    model.language_model.embed_tokens.requires_grad_(False)
    model.visual.requires_grad_(tc.train_vision)
    if tc.train_llm == "frozen":
        model.language_model.requires_grad_(False)
        model.language_model.embed_tokens.requires_grad_(False)
    elif tc.train_llm == "lora":
        model.language_model.requires_grad_(False)
        from peft import LoraConfig, inject_adapter_in_model

        lora = LoraConfig(
            r=tc.lora_r, lora_alpha=tc.lora_alpha, lora_dropout=tc.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model.language_model = inject_adapter_in_model(lora, model.language_model)
    elif tc.train_llm != "full":
        raise ValueError(f"bad train_llm {tc.train_llm}")

    if model.exp12_enabled:
        # Legacy latent modules are kept for checkpoint/config compatibility but
        # must not enter an EXP-12 optimizer when their objectives are disabled.
        model.mask_embed.requires_grad_(False)
        model.reg_head.requires_grad_(False)
        if model.mtp_heads is not None:
            model.mtp_heads.requires_grad_(False)
        if hasattr(model, "attn_pool"):
            model.attn_pool.requires_grad_(False)
        model.assert_exp12_frozen_visual()

    if tc.gradient_checkpointing:
        if tc.train_vision:
            model.visual.gradient_checkpointing_enable()
        model.language_model.gradient_checkpointing_enable()
    return model
