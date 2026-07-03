"""Regression head and multi-step (MTP) latent prediction heads. All lightweight 2-layer MLPs."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MTPHeads(nn.Module):
    """k independent heads; head j-1 regresses h_{t+j} from the hidden state at position t."""

    def __init__(self, dim: int, k: int, hidden: int = 0):
        super().__init__()
        self.k = k
        self.heads = nn.ModuleList([MLPHead(dim, hidden) for _ in range(k)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [head(x) for head in self.heads]
