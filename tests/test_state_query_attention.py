import torch

from jepa_vlm.data.state_sampler import sample_state_pairs

from .helpers import fake_exp12_model


def test_query_student_sequence_never_contains_target_tokens():
    cfg, model = fake_exp12_model()
    source = torch.randn(1, 16, 4, 16)
    target = torch.full_like(source, 9999.0).detach()
    pairs = sample_state_pairs(source, target, 2, torch.tensor([True]))
    loss, metrics = model._run_state_transition(pairs, (2, 2), "query")
    inputs = model.language_model.last_inputs
    assert inputs.shape[1] == 4 + 1 + 4
    assert not torch.any(inputs == 9999.0)
    assert pairs.target.requires_grad is False
    assert torch.isfinite(loss)
    assert "state/persistence_ratio" in metrics
