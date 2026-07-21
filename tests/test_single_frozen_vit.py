from jepa_vlm.train import make_optimizer, parameter_audit

from .helpers import fake_exp12_model


def test_single_physical_visual_and_no_frozen_optimizer_params():
    cfg, model = fake_exp12_model()
    model.assert_exp12_frozen_visual()
    optimizer = make_optimizer(model, cfg)
    audit = parameter_audit(model, optimizer, cfg)
    assert audit["physical_visual_module_count"] == 1
    assert audit["visual_parameters_in_optimizer"] == 0
    assert audit["frozen_parameters_in_optimizer"] == 0
    assert audit["frozen_vit_parameters"] > 0
    assert audit["frozen_merger_parameters"] > 0
