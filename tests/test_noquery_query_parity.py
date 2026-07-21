import torch

from jepa_vlm.data.state_sampler import sample_state_pairs

from .helpers import fake_exp12_model


def test_noquery_and_query_share_source_target_horizon_and_head_spec():
    query_cfg, query_model = fake_exp12_model("a1_query_k4")
    noquery_cfg, noquery_model = fake_exp12_model("a1_query_k4")
    noquery_cfg.model.state_predictor_mode = "no_query"
    states = torch.randn(2, 16, 4, 16)
    targets = torch.randn_like(states).detach()
    q_pairs = sample_state_pairs(states, targets, 2, torch.tensor([True, True]))
    n_pairs = sample_state_pairs(states, targets, 2, torch.tensor([True, True]))
    assert torch.equal(q_pairs.source, n_pairs.source)
    assert torch.equal(q_pairs.target, n_pairs.target)
    assert query_cfg.train.state_horizon_units == noquery_cfg.train.state_horizon_units
    q_head = sum(p.numel() for p in query_model.state_transition_head.parameters())
    n_head = sum(p.numel() for p in noquery_model.state_transition_head.parameters())
    assert q_head == n_head
