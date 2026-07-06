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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .heads import MLPHead, MTPHeads
from .masking import sample_token_mask
from .pooling import AttnPool, avg_pool_frames

POOL_SIDE = 2  # pooled grid side -> tokens_per_frame = POOL_SIDE**2


@dataclass
class JepaOutput:
    loss: torch.Tensor | None = None
    reg_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    ce_loss: torch.Tensor | None = None
    metrics: dict | None = None
    hidden_states: tuple | None = None
    token_mask: torch.Tensor | None = None
    target: torch.Tensor | None = None


def visual_only_position_ids(B: int, T: int, device) -> torch.Tensor:
    """MRoPE ids for a pure visual sequence of T frames x 4 pooled tokens (native convention:
    each frame is an independent 1 x 2 x 2 grid, `current_pos` advances by 2 per frame).

    Returns (4, B, L): row 0 = text positions (arange, used for causal-mask bookkeeping),
    rows 1..3 = (t, h, w).
    """
    P = POOL_SIDE * POOL_SIDE
    L = T * P
    starts = torch.arange(T, device=device) * POOL_SIDE          # frame start offsets
    t_ids = starts.repeat_interleave(P)
    h_ids = (starts[:, None] + torch.tensor([0, 0, 1, 1], device=device)[None, :]).reshape(-1)
    w_ids = (starts[:, None] + torch.tensor([0, 1, 0, 1], device=device)[None, :]).reshape(-1)
    text_row = torch.arange(L, device=device)
    pos = torch.stack([text_row, t_ids, h_ids, w_ids], dim=0)    # (4, L)
    return pos[:, None, :].expand(-1, B, -1).contiguous()


def mixed_position_ids(input_ids: torch.Tensor, video_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    """MRoPE ids for text + video sequences (Phase B). Each sample must contain exactly
    num_frames * 4 video placeholder tokens (contiguous per frame). Returns (4, B, L)."""
    B, L = input_ids.shape
    P = POOL_SIDE * POOL_SIDE
    device = input_ids.device
    pos = torch.zeros(4, B, L, dtype=torch.long, device=device)
    frame_offsets = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]], device=device)  # (3, P) t,h,w
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
                pos[0, b, i:i + P] = torch.arange(P, device=device) + cur
                pos[1:, b, i:i + P] = frame_offsets + cur
                cur += POOL_SIDE
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
        self.mask_embed = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.mask_embed, std=0.02)

        self.pooling = mc.pooling
        if mc.pooling == "attn":
            self.attn_pool = AttnPool(D, num_queries=mc.tokens_per_frame)
        assert mc.tokens_per_frame == POOL_SIDE * POOL_SIDE, "only 2x2 pooling supported"

        self.reg_head = MLPHead(D, mc.reg_head_hidden)
        self.mtp_heads = MTPHeads(D, mc.mtp_k, mc.reg_head_hidden) if mc.mtp_enabled else None
        self.target_norm = nn.LayerNorm(D, elementwise_affine=False)

        if mc.bidirectional_visual and mc.mtp_enabled:
            raise ValueError("bidirectional_visual sees future frames -> MTP is trivial; "
                             "set model.mtp_enabled=false for this ablation")

    # ------------------------------------------------------------------ vision
    def encode_video(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """pixel_values: (B, S, patch_dim) patchified videos (identical grids across batch).
        grid_thw: (3,) or (B, 3) in patch units.

        Returns h: (B, T, P, D) pooled tokens, deepstack: list of (B, T, P, D) (pooled, avg)."""
        B = pixel_values.shape[0]
        grid_thw = grid_thw.to(pixel_values.device)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw[None, :].expand(B, -1)
        T, Hp, Wp = (int(x) for x in grid_thw[0])
        Hm, Wm = Hp // 2, Wp // 2
        flat = pixel_values.reshape(-1, pixel_values.shape[-1])
        out = self.visual(flat, grid_thw)
        merged = out.pooler_output.reshape(B, T, Hm, Wm, -1)
        if self.pooling == "attn":
            h = self.attn_pool(merged)
        else:
            h = avg_pool_frames(merged, POOL_SIDE)
        deepstack = []
        if self.cfg.model.use_deepstack:
            for feat in out.deepstack_features:
                deepstack.append(avg_pool_frames(feat.reshape(B, T, Hm, Wm, -1), POOL_SIDE))
        return h, deepstack

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
    ) -> JepaOutput:
        mc = self.cfg.model
        h, deepstack = self.encode_video(pixel_values, grid_thw)
        B, T, P, D = h.shape

        target = self.target_norm(h.float()).detach()

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
                h_in, deepstack, target, token_mask if use_mask else None, output_hidden_states
            )
        return self._forward_with_text(
            h_in, deepstack, target, token_mask if use_mask else None,
            input_ids, attention_mask, labels, output_hidden_states,
        )

    def _forward_visual_only(self, h_in, deepstack, target, token_mask, output_hidden_states):
        mc = self.cfg.model
        B, T, P, D = h_in.shape
        L = T * P
        device = h_in.device
        inputs_embeds = h_in.reshape(B, L, D)
        position_ids = visual_only_position_ids(B, T, device)
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
        loss, reg_loss, mtp_loss, metrics = self._compute_losses(hidden, target, token_mask)
        return JepaOutput(
            loss=loss, reg_loss=reg_loss, mtp_loss=mtp_loss, metrics=metrics,
            hidden_states=getattr(outputs, "hidden_states", None),
            token_mask=token_mask, target=target,
        )

    def _forward_with_text(self, h_in, deepstack, target, token_mask,
                           input_ids, attention_mask, labels, output_hidden_states):
        B, T, P, D = h_in.shape
        device = h_in.device
        video_token_id = self.hf_config.video_token_id
        video_mask = input_ids == video_token_id  # (B, L)
        assert int(video_mask.sum()) == B * T * P, "each sample needs exactly T*P video tokens"

        inputs_embeds = self.language_model.embed_tokens(input_ids).clone()
        inputs_embeds[video_mask] = h_in.reshape(-1, D).to(inputs_embeds.dtype)
        position_ids = mixed_position_ids(input_ids, video_mask, T)
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
        loss, reg_loss, mtp_loss, metrics = self._compute_losses(hidden, target, token_mask)

        ce_loss = None
        if labels is not None:
            logits = self.lm_head(seq_hidden)
            ce_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            metrics["ce_loss"] = float(ce_loss.detach())
            lam = self.cfg.train.lambda_reg
            loss = ce_loss + lam * loss if loss is not None else ce_loss
        return JepaOutput(
            loss=loss, reg_loss=reg_loss, mtp_loss=mtp_loss, ce_loss=ce_loss, metrics=metrics,
            hidden_states=getattr(outputs, "hidden_states", None),
            token_mask=token_mask, target=target,
        )

    # ------------------------------------------------------------------ losses & monitors
    def _compute_losses(self, hidden, target, token_mask):
        mc = self.cfg.model
        B, T, P, D = hidden.shape
        pred = self.reg_head(hidden).float()

        if mc.mask_variant == "v2.1" and token_mask is not None:
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

        loss = reg_loss if mtp_loss is None else reg_loss + mtp_loss

        metrics = {
            "reg_loss": float(reg_loss.detach()),
            **mtp_metrics,
            **self._monitors(target, token_mask),
        }
        if mtp_loss is not None:
            metrics["mtp_loss"] = float(mtp_loss.detach())
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

        if token_mask is not None:
            frame_masked = token_mask.all(dim=-1)  # (B, T)
            errs = []
            for bi in range(B):
                masked_idx = frame_masked[bi].nonzero().flatten()
                free_idx = (~frame_masked[bi]).nonzero().flatten()
                if len(masked_idx) == 0 or len(free_idx) == 0:
                    continue
                for t in masked_idx.tolist():
                    past = free_idx[free_idx < t]
                    src = int(past.max()) if len(past) else int(free_idx[free_idx > t].min())
                    errs.append(F.mse_loss(target[bi, src], target[bi, t]))
            if errs:
                out["copy_mse"] = float(torch.stack(errs).mean())
        return out

    # ------------------------------------------------------------------ probes
    @torch.no_grad()
    def extract_features(self, pixel_values, grid_thw, layers: list[int]) -> dict[int, torch.Tensor]:
        """No-mask forward; returns {layer_idx: (B, T, P, D)} hidden states at visual positions.
        layer_idx indexes the recorded decoder-layer outputs (negative ok)."""
        out = self.forward(pixel_values, grid_thw, disable_mask=True, output_hidden_states=True)
        hs = out.hidden_states
        B = pixel_values.shape[0]
        feats = {}
        for li in layers:
            x = hs[li]
            feats[li] = x.reshape(B, -1, POOL_SIDE * POOL_SIDE, x.shape[-1])
        return feats

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

    if tc.gradient_checkpointing:
        model.visual.gradient_checkpointing_enable()
        model.language_model.gradient_checkpointing_enable()
    return model
