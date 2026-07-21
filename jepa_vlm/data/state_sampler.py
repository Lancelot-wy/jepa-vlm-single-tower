"""Aligned current/future state sampling for EXP-12."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StatePairs:
    source: torch.Tensor
    source_target_space: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    source_units: torch.Tensor
    target_units: torch.Tensor


def sample_state_pairs(
    states: torch.Tensor,
    target_states: torch.Tensor,
    horizon_units: int,
    sample_eligible: torch.Tensor | None = None,
) -> StatePairs:
    """Create every aligned source/target pair at one fixed horizon.

    `states` are LLM inputs while `target_states` are normalized frozen merger
    outputs.  Invalid short/duplicate samples remain represented with a false
    validity mask so all DDP ranks execute the same model/collective structure.
    """
    if states.ndim != 4 or target_states.shape != states.shape:
        raise ValueError("states and target_states must have matching [B,T,K,D] shapes")
    B, T, K, D = states.shape
    count = T - horizon_units
    if horizon_units < 1 or count <= 0:
        raise ValueError("horizon must leave at least one source-target pair")
    source_units = torch.arange(count, device=states.device)
    target_units = source_units + horizon_units
    source = states[:, :count].reshape(B * count, K, D)
    source_target = target_states[:, :count].reshape(B * count, K, D)
    target = target_states[:, horizon_units:].reshape(B * count, K, D).detach()
    if sample_eligible is None:
        eligible = torch.ones(B, dtype=torch.bool, device=states.device)
    else:
        eligible = sample_eligible.to(device=states.device, dtype=torch.bool).reshape(B)
    valid = eligible[:, None, None].expand(B, count, K).reshape(B * count, K)
    if target.requires_grad:
        raise AssertionError("future target must be detached")
    return StatePairs(
        source=source,
        source_target_space=source_target,
        target=target,
        valid=valid,
        source_units=source_units,
        target_units=target_units,
    )
