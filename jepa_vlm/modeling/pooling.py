"""Per-frame pooling of merged vision tokens down to a small fixed token count (default 2x2=4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
