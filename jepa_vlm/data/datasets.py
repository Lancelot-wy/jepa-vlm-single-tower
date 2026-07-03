"""Datasets over the unified manifest format.

Manifest = jsonl, one clip per line:
  {"video": "rel/path.mp4", "label": 12, "label_name": "...", "start": null, "end": null,
   "flow": 3.2, "duration": 4.1}
Only "video" is required. `label` feeds the linear probes; `flow` (mean optical-flow
magnitude, see scripts/compute_flow.py) enables static-clip filtering; `start`/`end`
crop a segment (Ego4D / EPIC style long videos).

Phase B QA manifest: {"video": ..., "question": ..., "answer": ...}.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_io import decode_frames, patchify, resize_center_crop


def load_manifest(path: str, min_flow: float = 0.0) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if min_flow > 0 and d.get("flow") is not None and d["flow"] < min_flow:
                continue
            items.append(d)
    if not items:
        raise ValueError(f"empty manifest {path} (min_flow={min_flow})")
    return items


class ManifestVideoDataset(Dataset):
    """Phase A dataset: video -> (pixel_values, grid_thw, label).

    temporal_transform:
      none            - as-is (training / class probes)
      shuffle|reverse - always apply (feature extraction for temporal probes)
      random_shuffle | random_reverse - 50/50 apply; label overridden with 0/1
                        (builds balanced temporal-probe sets on the fly)
    """

    def __init__(
        self,
        manifest: str,
        data_root: str = "",
        num_frames: int = 16,
        sample_fps: float = 2.0,
        frame_sampling: str = "fps_or_uniform",
        frame_size: int = 256,
        duplicate_frames: bool = True,
        min_flow: float = 0.0,
        training: bool = True,
        temporal_transform: str = "none",
        seed: int = 0,
    ):
        self.items = load_manifest(manifest, min_flow)
        self.data_root = data_root
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.frame_sampling = frame_sampling
        self.frame_size = frame_size
        self.duplicate_frames = duplicate_frames
        self.training = training
        self.temporal_transform = temporal_transform
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def _apply_temporal(self, frames: np.ndarray, rng: np.random.Generator, label: int):
        tt = self.temporal_transform
        if tt == "none":
            return frames, label
        if tt in ("random_shuffle", "random_reverse"):
            apply = bool(rng.integers(0, 2))
            label = int(apply)
            tt = tt.removeprefix("random_") if apply else "none"
        if tt == "reverse":
            frames = frames[::-1].copy()
        elif tt == "shuffle":
            perm = rng.permutation(len(frames))
            while (perm == np.arange(len(frames))).all():
                perm = rng.permutation(len(frames))
            frames = frames[perm].copy()
        return frames, label

    def __getitem__(self, i: int):
        it = self.items[i]
        rng = np.random.default_rng(None if self.training else self.seed * 100003 + i)
        path = os.path.join(self.data_root, it["video"]) if self.data_root else it["video"]
        frames = decode_frames(
            path, self.num_frames, self.sample_fps, self.frame_sampling,
            start=it.get("start"), end=it.get("end"),
            random_offset=self.training, rng=rng,
        )
        label = int(it.get("label", -1) if it.get("label") is not None else -1)
        frames, label = self._apply_temporal(frames, rng, label)
        pixel_values, grid = patchify(
            resize_center_crop(frames, self.frame_size), self.duplicate_frames
        )
        return {"pixel_values": pixel_values, "grid_thw": grid, "label": label, "index": i}


def collate_visual(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "grid_thw": batch[0]["grid_thw"],
        "labels_cls": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------- Phase B
class QAVideoDataset(ManifestVideoDataset):
    """Phase B: (video, question, answer). With prob `temporal_qa_ratio` the QA pair is
    replaced by an on-the-fly temporal-order QA (shuffled/reversed/normal frames)."""

    def __init__(self, *args, temporal_qa_ratio: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_qa_ratio = temporal_qa_ratio

    TEMPORAL_Q = "Are the frames of this video shown in the correct temporal order? Answer yes or no."

    def __getitem__(self, i: int):
        it = self.items[i]
        rng = np.random.default_rng(None if self.training else self.seed * 100003 + i)
        path = os.path.join(self.data_root, it["video"]) if self.data_root else it["video"]
        frames = decode_frames(
            path, self.num_frames, self.sample_fps, self.frame_sampling,
            start=it.get("start"), end=it.get("end"),
            random_offset=self.training, rng=rng,
        )
        question, answer = it.get("question", ""), it.get("answer", "")
        if self.training and rng.random() < self.temporal_qa_ratio:
            corrupt = bool(rng.integers(0, 2))
            if corrupt:
                if rng.integers(0, 2):
                    frames = frames[::-1].copy()
                else:
                    perm = rng.permutation(len(frames))
                    frames = frames[perm].copy()
            question, answer = self.TEMPORAL_Q, ("no" if corrupt else "yes")
        pixel_values, grid = patchify(
            resize_center_crop(frames, self.frame_size), self.duplicate_frames
        )
        return {"pixel_values": pixel_values, "grid_thw": grid,
                "question": question, "answer": answer}


class QACollator:
    """Builds Qwen chat-format input_ids with T*P video placeholder tokens; labels cover
    only the answer tokens. Works with the HF tokenizer (or any object exposing
    encode/eos_token_id/pad_token_id and the qwen special token ids in `cfg_ids`)."""

    def __init__(self, tokenizer, cfg_ids: dict, tokens_per_clip: int, max_len: int = 256):
        self.tok = tokenizer
        self.ids = cfg_ids  # video_token_id / vision_start_token_id / vision_end_token_id
        self.tokens_per_clip = tokens_per_clip
        self.max_len = max_len

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def __call__(self, batch: list[dict]) -> dict:
        seqs, labels = [], []
        for b in batch:
            pre = self._encode("<|im_start|>user\n")
            vid = [self.ids["vision_start_token_id"]] + \
                  [self.ids["video_token_id"]] * self.tokens_per_clip + \
                  [self.ids["vision_end_token_id"]]
            q = self._encode(b["question"] + "<|im_end|>\n<|im_start|>assistant\n")
            a = self._encode(b["answer"] + "<|im_end|>")
            ids = (pre + vid + q + a)[: self.max_len + self.tokens_per_clip]
            lab = [-100] * (len(pre) + len(vid) + len(q)) + a
            lab = lab[: len(ids)]
            seqs.append(ids)
            labels.append(lab)
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        lab_t = torch.full((len(seqs), L), -100, dtype=torch.long)
        attn = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            input_ids[i, : len(s)] = torch.tensor(s)
            lab_t[i, : len(l)] = torch.tensor(l)
            attn[i, : len(s)] = 1
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "grid_thw": batch[0]["grid_thw"],
            "input_ids": input_ids, "attention_mask": attn, "labels": lab_t,
        }
