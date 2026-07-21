"""Observation/Event queries and transition heads for EXP-12."""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import MLPHead


class StateQueryBuilder(nn.Module):
    """K learned queries with aligned row, column, and horizon embeddings."""

    def __init__(
        self,
        dim: int,
        query_count: int,
        num_horizons: int,
        position_encoding: bool = True,
    ):
        super().__init__()
        self.query_count = query_count
        self.position_encoding = position_encoding
        self.learned_query = nn.Parameter(torch.empty(query_count, dim))
        self.row_embedding = nn.Embedding(query_count, dim)
        self.col_embedding = nn.Embedding(query_count, dim)
        self.horizon_embedding = nn.Embedding(num_horizons, dim)
        nn.init.normal_(self.learned_query, std=0.02)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)
        nn.init.normal_(self.horizon_embedding.weight, std=0.02)

    def forward(
        self,
        batch_size: int,
        pooled_grid: tuple[int, int],
        horizon_id: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        rows, cols = pooled_grid
        if rows * cols != self.query_count:
            raise ValueError("query count and pooled spatial grid do not match")
        query = self.learned_query.to(device=device, dtype=dtype)
        if self.position_encoding:
            row_ids = torch.arange(rows, device=device).repeat_interleave(cols)
            col_ids = torch.arange(cols, device=device).repeat(rows)
            query = query + self.row_embedding(row_ids).to(dtype)
            query = query + self.col_embedding(col_ids).to(dtype)
        horizon = self.horizon_embedding(
            torch.tensor(horizon_id, device=device)
        ).to(dtype)
        query = query + horizon
        return query.unsqueeze(0).expand(batch_size, -1, -1)


class TransitionHead(MLPHead):
    """Named two-layer MLP shared by query/no-query specifications."""


def horizon_embedding_id(value: float, supported: tuple[float, ...] | list[float]) -> int:
    for index, candidate in enumerate(supported):
        if abs(float(candidate) - float(value)) < 1e-6:
            return index
    raise ValueError(f"horizon {value} is not in configured horizon values {list(supported)}")


def query_position_ids(
    batch_size: int,
    pooled_grid: tuple[int, int],
    sequence_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """MRoPE IDs for K row-major query positions aligned to target geometry."""
    rows, cols = pooled_grid
    k = rows * cols
    row = torch.arange(rows, device=device).repeat_interleave(cols) + sequence_offset
    col = torch.arange(cols, device=device).repeat(rows) + sequence_offset
    temporal = torch.full((k,), sequence_offset, dtype=torch.long, device=device)
    text = torch.arange(k, device=device) + sequence_offset
    pos = torch.stack([text, temporal, row, col], dim=0)
    return pos[:, None, :].expand(-1, batch_size, -1).contiguous()
