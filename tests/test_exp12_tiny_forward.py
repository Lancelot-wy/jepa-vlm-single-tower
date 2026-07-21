import torch

from jepa_vlm.config import load_config
from jepa_vlm.data.datasets import QACollator
from jepa_vlm.data.video_io import patchify
from jepa_vlm.modeling.model import build_model
from jepa_vlm.train import ByteTokenizer


def test_tiny_qwen_end_to_end_query_forward_backward():
    cfg = load_config("configs/orca_token_sweep/a1_query_k4.yaml")
    cfg.model.tiny_config = True
    cfg.model.pretrained = "tiny"
    cfg.model.dtype = "float32"
    cfg.model.frame_size = 64
    cfg.train.gradient_checkpointing = False
    model = build_model(cfg)
    frames = torch.linspace(0, 1, 32)[:, None, None, None].expand(32, 3, 64, 64).contiguous()
    pixels, grid = patchify(frames, duplicate_frames=False, temporal_patch_size=2)
    ids = {key: getattr(model.hf_config, key) for key in (
        "video_token_id", "vision_start_token_id", "vision_end_token_id"
    )}
    collator = QACollator(ByteTokenizer(), ids, 16 * 4, 256)
    batch = collator([{
        "pixel_values": pixels, "grid_thw": grid,
        "question": "What changes?", "answer": "motion", "index": 0,
        "video_stats": {"state_eligible": True},
    }])
    output = model(
        pixel_values=batch["pixel_values"], grid_thw=batch["grid_thw"],
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        labels=batch["labels"], state_eligible=torch.tensor([True]),
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.ce_loss)
    assert torch.isfinite(output.state_loss)
    assert output.metrics["model/visual_tokens_per_unit"] == 4
    assert output.metrics["model/deepstack_token_count"] == 4
    assert not any(parameter.grad is not None for parameter in model.visual.parameters())
    model.eval()
    with torch.no_grad():
        features = model.extract_features(
            batch["pixel_values"], batch["grid_thw"], layers=[-1]
        )[-1]
    assert features.shape[:3] == (1, 16, 4)
