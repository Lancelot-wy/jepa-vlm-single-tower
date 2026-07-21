import types

import torch

from .helpers import fake_exp12_model


def test_event_target_and_wrong_event_do_not_enter_student_llm():
    cfg, model = fake_exp12_model("a1_query_k4")
    cfg.model.state_predictor_mode = "observation_event_query"
    cfg.model.event_condition_enable = True
    # Add the event modules because the helper started from the query-only arm.
    from jepa_vlm.modeling.state_prediction import StateQueryBuilder
    model.event_query_builder = StateQueryBuilder(16, 4, 8, True)
    model.event_direction_embedding = torch.nn.Embedding(2, 16)

    source_states = torch.randn(1, 16, 4, 16)
    target_states = torch.full_like(source_states, 7777.0)
    negative_states = torch.full_like(source_states, -7777.0)
    calls = iter((source_states, target_states, negative_states))

    def fake_encode(self, pixels, grid):
        return next(calls), (2, 2)

    model._encode_frozen_temporal_units = types.MethodType(fake_encode, model)
    batch = {
        "source_pixel_values": torch.zeros(1, 1), "source_grid_thw": torch.tensor([1, 1, 1]),
        "target_pixel_values": torch.ones(1, 1), "target_grid_thw": torch.tensor([1, 1, 1]),
        "negative_pixel_values": torch.ones(1, 1) * 2,
        "negative_grid_thw": torch.tensor([1, 1, 1]),
        "source_inner_fraction": torch.tensor([0.5]),
        "target_inner_fraction": torch.tensor([0.5]),
        "condition_input_ids": torch.tensor([[10, 11]]),
        "condition_attention_mask": torch.tensor([[1, 1]]),
        "direction": torch.tensor([1]),
    }
    loss, metrics = model._run_event_transition(batch)
    student = model.language_model.last_inputs
    assert not torch.any(student == 7777.0)
    assert not torch.any(student == -7777.0)
    assert torch.isfinite(loss)
    assert "event/same_video_negative_cos" in metrics
