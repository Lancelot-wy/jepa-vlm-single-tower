from jepa_vlm.config import (
    load_config,
    resolved_raw_num_frames,
    resolved_temporal_units,
    resolved_visual_tokens,
)


ARMS = (
    "a0_ce_k4", "a1_query_k4", "a2_ce_k16",
    "a3_query_k16", "a4_ce_k64", "a5_query_k64",
)


def test_six_configs_and_fixed_contract():
    configs = [load_config(f"configs/orca_token_sweep/{name}.yaml") for name in ARMS]
    assert [resolved_visual_tokens(cfg) for cfg in configs] == [4, 4, 16, 16, 64, 64]
    assert [cfg.model.state_predictor_mode for cfg in configs] == [
        "none", "query", "none", "query", "none", "query"
    ]
    for cfg in configs:
        assert resolved_raw_num_frames(cfg) == 32
        assert resolved_temporal_units(cfg) == 16
        assert cfg.train.state_horizon_units == 2
        assert cfg.train.state_horizon_seconds == 1.0
        assert not cfg.train.train_vision
        assert cfg.train.train_llm == "full"
        assert cfg.train.max_steps == 800
        assert cfg.model.event_condition_enable is False
        assert cfg.model.beat_copy_loss_weight == 0.0
        assert cfg.model.random_mask_ratio == 0.0


def test_only_scientific_arm_variables_change():
    configs = [load_config(f"configs/orca_token_sweep/{name}.yaml") for name in ARMS]
    normalized = []
    for cfg in configs:
        value = cfg.to_dict()
        value["model"]["visual_tokens_per_unit"] = "K"
        value["model"]["state_predictor_mode"] = "MODE"
        value["train"]["output_dir"] = "OUTPUT"
        normalized.append(value)
    assert all(value == normalized[0] for value in normalized[1:])
