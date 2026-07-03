"""Frame-level (tube) and token-level (patch, negative control) mask sampling."""

from __future__ import annotations

import torch


def sample_frame_mask(
    batch_size: int,
    num_frames: int,
    ratio: float,
    max_run: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Tube masking over whole latent frames: random mix of contiguous runs and scattered frames.

    Returns bool (B, T), True = masked. Guarantees >=1 unmasked and (ratio>0) >=1 masked frame.
    """
    n_mask = min(max(int(round(ratio * num_frames)), 1), num_frames - 1)
    mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
    for b in range(batch_size):
        remaining = n_mask
        attempts = 0
        while remaining > 0 and attempts < 100:
            run = int(torch.randint(1, min(max_run, remaining) + 1, (1,), generator=generator))
            start = int(torch.randint(0, num_frames, (1,), generator=generator))
            end = min(start + run, num_frames)
            newly = int((~mask[b, start:end]).sum())
            take = min(newly, remaining)
            if take > 0:
                idx = torch.arange(start, end)[~mask[b, start:end]][:take]
                mask[b, idx] = True
                remaining -= take
            attempts += 1
        if remaining > 0:  # fallback: scatter the remainder
            free = (~mask[b]).nonzero().flatten()
            perm = torch.randperm(len(free), generator=generator)[:remaining]
            mask[b, free[perm]] = True
    return mask


def sample_token_mask(
    batch_size: int,
    num_frames: int,
    tokens_per_frame: int,
    ratio: float,
    mode: str = "tube",
    max_run: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (token_mask (B,T,P) bool, frame_mask (B,T) bool).

    tube: whole frames masked (plan default). patch: random tokens across all frames
    (expected to fail per plan section 5.1 - mechanism ablation only).
    """
    if mode == "tube":
        frame_mask = sample_frame_mask(batch_size, num_frames, ratio, max_run, generator)
        token_mask = frame_mask[:, :, None].expand(-1, -1, tokens_per_frame).clone()
    elif mode == "patch":
        n_tok = num_frames * tokens_per_frame
        n_mask = min(max(int(round(ratio * n_tok)), 1), n_tok - 1)
        token_mask = torch.zeros(batch_size, n_tok, dtype=torch.bool)
        for b in range(batch_size):
            perm = torch.randperm(n_tok, generator=generator)[:n_mask]
            token_mask[b, perm] = True
        token_mask = token_mask.view(batch_size, num_frames, tokens_per_frame)
        frame_mask = token_mask.all(dim=-1)
    else:
        raise ValueError(f"unknown mask mode {mode}")
    return token_mask, frame_mask
