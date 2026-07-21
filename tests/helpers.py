from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from jepa_vlm.config import load_config
from jepa_vlm.modeling.model import JepaQwen3VL


class FakeVisual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.merger = nn.Linear(dim, dim)


class FakeLanguageModel(nn.Module):
    def __init__(self, dim: int, vocab: int = 256):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.last_inputs = None

    def forward(self, inputs_embeds, **kwargs):
        self.last_inputs = inputs_embeds.detach().clone()
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds), hidden_states=None)


def fake_exp12_model(config_name: str = "a1_query_k4", dim: int = 16):
    cfg = load_config(f"configs/orca_token_sweep/{config_name}.yaml")
    language = FakeLanguageModel(dim)
    visual = FakeVisual(dim)
    fake_hf = SimpleNamespace(
        model=SimpleNamespace(visual=visual, language_model=language),
        lm_head=nn.Linear(dim, 256, bias=False),
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=dim),
            video_token_id=8,
            vision_start_token_id=5,
            vision_end_token_id=6,
        ),
    )
    model = JepaQwen3VL(fake_hf, cfg)
    model.visual.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.language_model.embed_tokens.requires_grad_(False)
    model.mask_embed.requires_grad_(False)
    model.reg_head.requires_grad_(False)
    if model.mtp_heads is not None:
        model.mtp_heads.requires_grad_(False)
    return cfg, model
