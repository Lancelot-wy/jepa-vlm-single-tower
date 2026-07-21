import torch

from jepa_vlm.modeling.state_loss import DistributedRunningCenter, compute_state_objective


def objective(pred, target, current):
    center = DistributedRunningCenter(target.shape[-1], 0.99)
    valid = torch.ones(target.shape[:-1], dtype=torch.bool)
    return compute_state_objective(
        pred, target.detach(), current, valid, center,
        dynamic_threshold=0.05, dynamic_weighting=True,
        beat_copy_loss_weight=0.0, beat_copy_margin=0.05,
        update_center=True,
    )


def test_perfect_future_prediction_beats_copy():
    current = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
    result = objective(target.clone().requires_grad_(), target, current)
    assert result.metrics["state/persistence_ratio"] < 0.01


def test_copy_prediction_has_ratio_one():
    current = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    result = objective(current.clone().requires_grad_(), target, current)
    assert abs(result.metrics["state/persistence_ratio"] - 1.0) < 1e-5
