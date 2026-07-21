"""Spatially structured pooling of merged vision tokens.

The legacy helper remains for historical checkpoints.  EXP-12 uses
``SpatialVisualTokenPooler`` so main and DeepStack features share the same
aspect-aware HxW -> pooled_h x pooled_w mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_pooled_grid(tokens: int, input_h: int, input_w: int) -> tuple[int, int]:
    """Choose an exact factorization of K closest to the input aspect ratio.

    Ties prefer the more balanced grid.  Square 8x8 merger grids therefore map
    K=4/16/64 to 2x2, 4x4, and 8x8 respectively.  Token order remains row-major.
    """
    if tokens < 1 or input_h < 1 or input_w < 1:
        raise ValueError("tokens and input grid dimensions must be positive")
    aspect = input_w / input_h
    candidates = []
    for h in range(1, tokens + 1):
        if tokens % h:
            continue
        w = tokens // h
        aspect_error = abs(torch.log(torch.tensor((w / h) / aspect)).item())
        balance = abs(w - h)
        candidates.append((aspect_error, balance, h, w))
    _, _, out_h, out_w = min(candidates)
    return out_h, out_w


class SpatialVisualTokenPooler(nn.Module):
    """Adaptive spatial average pooling with an explicit row-major K-token grid."""

    def __init__(self, tokens_per_unit: int):
        super().__init__()
        self.tokens_per_unit = tokens_per_unit

    def output_grid(self, input_h: int, input_w: int) -> tuple[int, int]:
        return resolve_pooled_grid(self.tokens_per_unit, input_h, input_w)

    def forward(
        self, x: torch.Tensor, output_grid: tuple[int, int] | None = None
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Pool `[B,T,H,W,D]` to `[B,T,K,D]` plus the resolved HxW grid."""
        if x.ndim != 5:
            raise ValueError(f"expected [B,T,H,W,D], got {tuple(x.shape)}")
        B, T, H, W, D = x.shape
        out_h, out_w = output_grid or self.output_grid(H, W)
        if out_h * out_w != self.tokens_per_unit:
            raise ValueError("pooled grid does not contain tokens_per_unit positions")
        y = x.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W)
        y = F.adaptive_avg_pool2d(y, (out_h, out_w))
        # Flattening H then W is the required row-major spatial token order.
        y = y.reshape(B, T, D, out_h * out_w).permute(0, 1, 3, 2).contiguous()
        if y.shape[-2] != self.tokens_per_unit:
            raise AssertionError("visual pooling emitted the wrong token count")
        return y, (out_h, out_w)


def avg_pool_frames(x: torch.Tensor, out_side: int = 2) -> torch.Tensor:
    """x: (B, T, Hm, Wm, D) merged tokens per latent frame -> (B, T, out_side*out_side, D)."""
    B, T, Hm, Wm, D = x.shape
    x = x.permute(0, 1, 4, 2, 3).reshape(B * T, D, Hm, Wm)
    x = F.adaptive_avg_pool2d(x, out_side)
    x = x.reshape(B, T, D, out_side * out_side).permute(0, 1, 3, 2)
    return x.contiguous()


class AttnPool(nn.Module):
    """Learned attention pooling: `num_queries` queries attend over each frame's merged tokens.

    Optional alternative to parameter-free avg pooling (config: model.pooling = "attn").
    Note the regression target is produced by the same pooling, so extra capacity here
    also shapes the target space; avg pooling is the safer default.
    """

    def __init__(self, dim: int, num_queries: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, Hm, Wm, D) -> (B, T, num_queries, D)."""
        B, T, Hm, Wm, D = x.shape
        n = Hm * Wm
        h = self.num_heads
        kv = self.kv(x.reshape(B * T, n, D))
        k, v = kv.chunk(2, dim=-1)
        q = self.query.unsqueeze(0).expand(B * T, -1, -1).to(x.dtype)
        q = q.reshape(B * T, self.num_queries, h, D // h).transpose(1, 2)
        k = k.reshape(B * T, n, h, D // h).transpose(1, 2)
        v = v.reshape(B * T, n, h, D // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B * T, self.num_queries, D)
        return self.proj(o).reshape(B, T, self.num_queries, D)
