"""Distributed centered-cosine objective and persistence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DistributedRunningCenter(nn.Module):
    """FP32 running target center synchronized across every DDP rank."""

    def __init__(self, dim: int, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_center", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    def _apply(self, fn):
        super()._apply(fn)
        # `model.to(bfloat16)` must never down-cast the statistical center.
        self.running_center = self.running_center.float()
        return self

    @torch.no_grad()
    def update(self, target: torch.Tensor, valid: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        target_f = target.detach().float()
        mask = valid.to(device=target.device, dtype=torch.bool)
        selected = target_f[mask]
        local_sum = (
            selected.sum(dim=0) if selected.numel()
            else torch.zeros_like(self.running_center, device=target.device)
        )
        local_count = torch.tensor(
            float(selected.shape[0]), dtype=torch.float32, device=target.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        if enabled and local_count.item() > 0:
            batch_center = local_sum / local_count
            if not bool(self.initialized):
                self.running_center.copy_(batch_center)
                self.initialized.fill_(True)
            else:
                self.running_center.mul_(self.momentum).add_(
                    batch_center, alpha=1.0 - self.momentum
                )
            self.updates.add_(1)
        return self.running_center


@dataclass
class StateLossOutput:
    loss: torch.Tensor
    base_loss: torch.Tensor
    beat_copy_loss: torch.Tensor
    metrics: dict[str, float]


def _centered_normalize(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float() - center.float(), dim=-1, eps=1e-6)


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _metric(values: torch.Tensor, valid: torch.Tensor) -> float:
    return float(_masked_mean(values.detach().float(), valid).cpu())


def _effective_rank(target: torch.Tensor, valid: torch.Tensor, max_vectors: int = 128) -> float:
    x = target.detach().float()[valid]
    if x.shape[0] < 2:
        return 0.0
    if x.shape[0] > max_vectors:
        ids = torch.linspace(0, x.shape[0] - 1, max_vectors, device=x.device).long()
        x = x[ids]
    x = x - x.mean(dim=0, keepdim=True)
    # The sample-space Gram matrix is at most 128x128 even when D=2048.
    gram = x @ x.T
    # PyTorch MPS does not implement eigvalsh; this is a detached diagnostic,
    # so a tiny CPU fallback does not affect the training graph or CUDA path.
    if gram.device.type == "mps":
        gram = gram.cpu()
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    probs = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().cpu())


def _retrieval(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, cap: int = 512):
    p = pred.detach().float()[valid]
    t = target.detach().float()[valid]
    if p.shape[0] == 0:
        return 0.0, 0.0
    if p.shape[0] > cap:
        ids = torch.linspace(0, p.shape[0] - 1, cap, device=p.device).long()
        p, t = p[ids], t[ids]
    p, t = F.normalize(p, dim=-1), F.normalize(t, dim=-1)
    scores = p @ t.T
    ranking = scores.topk(min(5, scores.shape[1]), dim=1).indices
    gold = torch.arange(scores.shape[0], device=scores.device)[:, None]
    top1 = (ranking[:, :1] == gold).any(dim=1).float().mean()
    top5 = (ranking == gold).any(dim=1).float().mean()
    return float(top1.cpu()), float(top5.cpu())


def centered_cosine_distance(
    left: torch.Tensor, right: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    return 1.0 - (_centered_normalize(left, center) * _centered_normalize(right, center)).sum(-1)


def compute_state_objective(
    pred: torch.Tensor,
    target: torch.Tensor,
    current_state: torch.Tensor,
    valid: torch.Tensor,
    center_module: DistributedRunningCenter,
    *,
    dynamic_threshold: float,
    dynamic_weighting: bool,
    beat_copy_loss_weight: float,
    beat_copy_margin: float,
    update_center: bool,
    prefix: str = "state",
    negative_target: torch.Tensor | None = None,
) -> StateLossOutput:
    """Centered-cosine state loss with copy and shuffle baselines."""
    if pred.shape != target.shape or current_state.shape != target.shape:
        raise ValueError("pred, target, and current_state must share [N,K,D]")
    if valid.shape != target.shape[:-1]:
        raise ValueError("valid mask must have [N,K] shape")
    if target.requires_grad:
        raise AssertionError("state target must not require gradients")

    center = center_module.update(target, valid, enabled=update_center)
    pred_n = _centered_normalize(pred, center)
    target_n = _centered_normalize(target, center)
    current_n = _centered_normalize(current_state, center)
    per_token = 1.0 - (pred_n * target_n).sum(-1)
    copy_distance = 1.0 - (current_n * target_n).sum(-1)
    pred_distance = per_token
    dynamic_score = copy_distance.detach()
    if dynamic_weighting:
        dynamic_weight = (dynamic_score / dynamic_threshold).clamp(0.0, 1.0)
    else:
        dynamic_weight = torch.ones_like(dynamic_score)
    weights = dynamic_weight * valid.to(dynamic_weight.dtype)
    base_numerator = (weights * per_token).sum()
    beat = F.relu(pred_distance - copy_distance.detach() + beat_copy_margin)
    beat_numerator = (weights * beat).sum()

    # DDP averages gradients across ranks.  Scale each local numerator by
    # world_size/global_weight so the averaged gradient is the true global
    # weighted mean, including when some ranks contain only short/invalid clips.
    valid_float = valid.to(pred_distance.dtype)
    global_stats = torch.stack([
        base_numerator.detach(),
        beat_numerator.detach(),
        weights.sum().detach(),
        (pred_distance.detach() * valid_float).sum(),
        (copy_distance.detach() * valid_float).sum(),
        valid_float.sum(),
    ])
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    gradient_scale = world_size / global_stats[2].clamp_min(1.0)
    base_loss = base_numerator * gradient_scale
    beat_loss = beat_numerator * gradient_scale
    loss = base_loss + beat_copy_loss_weight * beat_loss

    raw_pred = F.normalize(pred.float(), dim=-1, eps=1e-6)
    raw_target = F.normalize(target.float(), dim=-1, eps=1e-6)
    raw_true = (raw_pred * raw_target).sum(-1)
    target_batch_shuffle = target.roll(1, dims=0)
    target_position_shuffle = target.roll(1, dims=1)
    raw_shuffle_batch = (raw_pred * F.normalize(target_batch_shuffle.float(), dim=-1)).sum(-1)
    raw_shuffle_position = (
        raw_pred * F.normalize(target_position_shuffle.float(), dim=-1)
    ).sum(-1)
    centered_true = (pred_n * target_n).sum(-1)
    centered_shuffle_batch = (
        pred_n * _centered_normalize(target_batch_shuffle, center)
    ).sum(-1)
    centered_shuffle_position = (
        pred_n * _centered_normalize(target_position_shuffle, center)
    ).sum(-1)
    centered_margin = centered_true - torch.maximum(
        centered_shuffle_batch, centered_shuffle_position
    )
    global_base_loss = global_stats[0] / global_stats[2].clamp_min(1.0)
    global_beat_loss = global_stats[1] / global_stats[2].clamp_min(1.0)
    persistence_ratio = global_stats[3] / global_stats[4].clamp_min(1e-8)
    dynamic_fraction = (dynamic_score >= dynamic_threshold).to(torch.float32)
    top1, top5 = _retrieval(pred_n, target_n, valid)
    metrics = {
        f"{prefix}/raw_true_cos": _metric(raw_true, valid),
        f"{prefix}/raw_shuffle_batch_cos": _metric(raw_shuffle_batch, valid),
        f"{prefix}/raw_shuffle_position_cos": _metric(raw_shuffle_position, valid),
        f"{prefix}/centered_true_cos": _metric(centered_true, valid),
        f"{prefix}/centered_shuffle_batch_cos": _metric(centered_shuffle_batch, valid),
        f"{prefix}/centered_shuffle_position_cos": _metric(centered_shuffle_position, valid),
        f"{prefix}/centered_margin": _metric(centered_margin, valid),
        f"{prefix}/target_std": float(target.detach().float()[valid].std(dim=0).mean().cpu())
            if valid.any() else 0.0,
        f"{prefix}/pred_std": float(pred.detach().float()[valid].std(dim=0).mean().cpu())
            if valid.sum() > 1 else 0.0,
        f"{prefix}/target_norm": _metric(target.detach().float().norm(dim=-1), valid),
        f"{prefix}/pred_norm": _metric(pred.detach().float().norm(dim=-1), valid),
        f"{prefix}/target_effective_rank": _effective_rank(target_n, valid),
        f"{prefix}/retrieval_top1": top1,
        f"{prefix}/retrieval_top5": top5,
        f"{prefix}/pred_distance": _metric(pred_distance, valid),
        f"{prefix}/copy_distance": _metric(copy_distance, valid),
        f"{prefix}/persistence_ratio": float(persistence_ratio.cpu()),
        f"{prefix}/dynamic_score": _metric(dynamic_score, valid),
        f"{prefix}/dynamic_sample_fraction": _metric(dynamic_fraction, valid),
        f"{prefix}/base_loss": float(global_base_loss.cpu()),
        f"{prefix}/beat_copy_loss": float(global_beat_loss.cpu()),
    }
    if negative_target is not None:
        negative_cos = (
            pred_n * _centered_normalize(negative_target.detach(), center)
        ).sum(-1)
        metrics[f"{prefix}/same_video_negative_cos"] = _metric(negative_cos, valid)
    return StateLossOutput(loss=loss, base_loss=base_loss, beat_copy_loss=beat_loss, metrics=metrics)
