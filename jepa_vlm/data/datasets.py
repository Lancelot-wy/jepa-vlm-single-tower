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

import contextlib
import json
import os
import signal
import sys
import threading

import numpy as np
import torch
from torch.utils.data import Dataset

from .video_io import decode_frames, patchify, resize_center_crop


class _DecodeTimeout(Exception):
    """Raised when building a single sample exceeds its wall-clock budget."""


@contextlib.contextmanager
def _hard_timeout(seconds: int):
    """Abort a hung decode with SIGALRM.

    A corrupt clip can make libav block forever inside ``av.open`` / decode
    without raising, which silently freezes a DataLoader worker and, in DDP,
    deadlocks every rank in the arm at the next collective.  ``signal.alarm``
    can only arm from the main thread, so in any other thread (or when the
    budget is non-positive) we no-op and rely on the plain exception path.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):
        raise _DecodeTimeout(f"decode exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def load_manifest(path: str, min_flow: float = 0.0) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # When a flow threshold is requested, a failed decode (flow=None)
            # must not enter training and fail later in a DataLoader worker.
            # Manifests without flow are still supported when min_flow == 0.
            if min_flow > 0 and (d.get("flow") is None or d["flow"] < min_flow):
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
        temporal_patch_size: int = 2,
        state_horizon_units: int = 1,
        min_flow: float = 0.0,
        training: bool = True,
        temporal_transform: str = "none",
        seed: int = 0,
        deterministic_order: bool = False,
    ):
        self.items = load_manifest(manifest, min_flow)
        self.data_root = data_root
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.frame_sampling = frame_sampling
        self.frame_size = frame_size
        self.duplicate_frames = duplicate_frames
        self.temporal_patch_size = temporal_patch_size
        self.state_horizon_units = state_horizon_units
        self.training = training
        self.temporal_transform = temporal_transform
        self.seed = seed
        self.deterministic_order = deterministic_order
        self.order = (
            np.random.default_rng(seed).permutation(len(self.items))
            if deterministic_order else np.arange(len(self.items))
        )

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

    _MAX_DECODE_RETRIES = 8
    _DECODE_TIMEOUT_SEC = 30

    def _resample_index(self, i: int, attempt: int) -> int:
        fb = np.random.default_rng(self.seed * 100003 + i * 1009 + attempt * 90173 + 1)
        return int(fb.integers(0, len(self.items)))

    def __getitem__(self, i: int):
        j = i
        for attempt in range(self._MAX_DECODE_RETRIES + 1):
            try:
                sample = self._build_sample(j)
                sample["video_stats"]["decode_retry_count"] = attempt
                return sample
            except Exception as exc:
                budget = getattr(self, "_decode_warn_left", 20)
                if budget > 0:
                    self._decode_warn_left = budget - 1
                    print(f"[dataset] skip undecodable idx={j}: {exc}", file=sys.stderr, flush=True)
                j = self._resample_index(i, attempt)
        raise RuntimeError(f"decode failed for {self._MAX_DECODE_RETRIES + 1} samples from index {i}")

    def _build_sample(self, i: int):
        manifest_i = int(self.order[i])
        it = self.items[manifest_i]
        # Keep video offset selection reproducible across paired arms.  The
        # DataLoader shuffle order may change, but a manifest row always maps
        # to one deterministic augmentation under a given train seed.
        rng = np.random.default_rng(self.seed * 100003 + manifest_i)
        path = os.path.join(self.data_root, it["video"]) if self.data_root else it["video"]
        with _hard_timeout(self._DECODE_TIMEOUT_SEC):
            frames, video_stats = decode_frames(
                path, self.num_frames, self.sample_fps, self.frame_sampling,
                start=it.get("start"), end=it.get("end"),
                random_offset=self.training, rng=rng,
                return_metadata=True, temporal_patch_size=self.temporal_patch_size,
                state_horizon_units=self.state_horizon_units,
            )
        label = int(it.get("label", -1) if it.get("label") is not None else -1)
        frames, label = self._apply_temporal(frames, rng, label)
        pixel_values, grid = patchify(
            resize_center_crop(frames, self.frame_size), self.duplicate_frames,
            self.temporal_patch_size,
        )
        video_stats["state_eligible"] = bool(
            video_stats["state_eligible"] and not self.duplicate_frames
        )
        video_stats["state_skipped_short"] = not video_stats["state_eligible"]
        video_stats["state_skipped_temporal_augmentation"] = False
        return {"pixel_values": pixel_values, "grid_thw": grid, "label": label,
                "index": manifest_i, "video_stats": video_stats}


def collate_visual(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "grid_thw": batch[0]["grid_thw"],
        "labels_cls": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    if batch and "video_stats" in batch[0]:
        out["state_eligible"] = torch.tensor(
            [bool(b["video_stats"]["state_eligible"]) for b in batch], dtype=torch.bool
        )
        out["video_stats"] = {
            key: torch.tensor([float(b["video_stats"][key]) for b in batch], dtype=torch.float32)
            for key in (
                "raw_frame_count", "unique_frame_count", "temporal_unit_count",
                "duplicate_adjacent_ratio", "effective_fps", "state_skipped_short",
                "state_skipped_temporal_augmentation",
                "nearest_frame_substitutions", "decode_retry_count",
            )
        }
    return out


# ---------------------------------------------------------------------- Phase B
class QAVideoDataset(ManifestVideoDataset):
    """Phase B: (video, question, answer). With prob `temporal_qa_ratio` the QA pair is
    replaced by an on-the-fly temporal augmentation sample. Template families:

      v1: order_yn only（帧序对不对，是非题）— EXP-04/08 的行为，保持不变
      v2: 均匀混合 5 个模板（对 TempCompass 的 direction/speed/order 弱项对症）:
          order_yn    帧序是非题（同 v1）
          order_mcq   帧序三选一：正常/倒放/打乱
          playback    正放 vs 倒放 二选一 MCQ（方向感知）
          speed       正常速 vs 2x 速 二选一 MCQ（帧距加倍模拟快放，标签自生成）
          pan         静帧滑窗合成"镜头左移/右移"二选一 MCQ（运动方向，标签自生成）
    所有标签来自我们自己的变换，零人工标注；两臂共享同一增广 -> 配对不受影响。"""

    V2_TEMPLATES = ("order_yn", "order_mcq", "playback", "speed", "pan")

    def __init__(self, *args, temporal_qa_ratio: float = 0.3,
                 temporal_qa_templates: str = "v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_qa_ratio = temporal_qa_ratio
        assert temporal_qa_templates in ("v1", "v2")
        self.temporal_qa_templates = temporal_qa_templates

    TEMPORAL_Q = "Are the frames of this video shown in the correct temporal order? Answer yes or no."

    # ---------------- template implementations (frames: uint8 (T,H,W,3)) ----------------
    def _tpl_order_yn(self, frames, rng):
        corrupt = bool(rng.integers(0, 2))
        if corrupt:
            if rng.integers(0, 2):
                frames = frames[::-1].copy()
            else:
                frames = frames[rng.permutation(len(frames))].copy()
        return frames, self.TEMPORAL_Q, ("no" if corrupt else "yes")

    def _tpl_order_mcq(self, frames, rng):
        kind = int(rng.integers(0, 3))  # 0 normal / 1 reversed / 2 shuffled
        if kind == 1:
            frames = frames[::-1].copy()
        elif kind == 2:
            frames = frames[rng.permutation(len(frames))].copy()
        q = ("Which best describes the frame order of this video?\nOptions:\n"
             "(A) correct chronological order\n(B) reversed\n(C) randomly shuffled\n"
             "Answer with the option's letter.")
        a = ["(A) correct chronological order", "(B) reversed", "(C) randomly shuffled"][kind]
        return frames, q, a

    def _tpl_playback(self, frames, rng):
        backward = bool(rng.integers(0, 2))
        if backward:
            frames = frames[::-1].copy()
        q = ("Is this video playing forward or backward?\nOptions:\n"
             "(A) forward\n(B) backward\nAnswer with the option's letter.")
        return frames, q, ("(B) backward" if backward else "(A) forward")

    def _tpl_speed(self, frames_fast_flag):
        fast = frames_fast_flag
        q = ("Is this video played at normal speed or fast (2x) speed?\nOptions:\n"
             "(A) normal speed\n(B) fast (2x) speed\nAnswer with the option's letter.")
        return q, ("(B) fast (2x) speed" if fast else "(A) normal speed")

    def _tpl_pan(self, frames, rng):
        """用中间帧合成滑窗序列：窗口从左到右或从右到左，标签 = 视野移动方向。"""
        base = frames[len(frames) // 2]
        H, W = base.shape[:2]
        cw = max(int(W * 0.6), 32)
        span = W - cw
        T = len(frames)
        offs = np.linspace(0, max(span, 1) - 1, T).astype(int)
        right = bool(rng.integers(0, 2))  # True: 视野向右移
        if not right:
            offs = offs[::-1]
        frames = np.stack([base[:, o:o + cw] for o in offs])
        q = ("Is the camera view moving left or right in this video?\nOptions:\n"
             "(A) left\n(B) right\nAnswer with the option's letter.")
        return frames, q, ("(B) right" if right else "(A) left")

    def _build_sample(self, i: int):
        manifest_i = int(self.order[i])
        it = self.items[manifest_i]
        # Do not use default_rng(None) here: it draws OS entropy independently
        # in CE and MTP arms, invalidating a same-seed paired comparison.
        rng = np.random.default_rng(self.seed * 100003 + manifest_i)
        path = os.path.join(self.data_root, it["video"]) if self.data_root else it["video"]

        # 先选模板再解码（speed 模板需要改采样帧率：帧距 x2 = 2 倍速）
        template = None
        if self.training and rng.random() < self.temporal_qa_ratio:
            if self.temporal_qa_templates == "v1":
                template = "order_yn"
            else:
                template = str(rng.choice(self.V2_TEMPLATES))
        fps = self.sample_fps
        speed_fast = False
        if template == "speed":
            speed_fast = bool(rng.integers(0, 2))
            if speed_fast:
                fps = self.sample_fps / 2.0

        with _hard_timeout(self._DECODE_TIMEOUT_SEC):
            frames, video_stats = decode_frames(
                path, self.num_frames, fps, self.frame_sampling,
                start=it.get("start"), end=it.get("end"),
                random_offset=self.training, rng=rng,
                return_metadata=True, temporal_patch_size=self.temporal_patch_size,
                state_horizon_units=self.state_horizon_units,
            )

        if template is None:
            question, answer = it.get("question", ""), it.get("answer", "")
        elif template == "order_yn":
            frames, question, answer = self._tpl_order_yn(frames, rng)
        elif template == "order_mcq":
            frames, question, answer = self._tpl_order_mcq(frames, rng)
        elif template == "playback":
            frames, question, answer = self._tpl_playback(frames, rng)
        elif template == "speed":
            question, answer = self._tpl_speed(speed_fast)
        elif template == "pan":
            frames, question, answer = self._tpl_pan(frames, rng)

        pixel_values, grid = patchify(
            resize_center_crop(frames, self.frame_size), self.duplicate_frames,
            self.temporal_patch_size,
        )
        real_future_eligible = bool(
            video_stats["state_eligible"] and not self.duplicate_frames
        )
        # EXP-11's CE recipe includes synthetic order/direction/speed questions.
        # Keep those examples for answer CE, but never call their transformed
        # sequence a real +1 s future target for the state objective.
        video_stats["state_eligible"] = bool(real_future_eligible and template is None)
        video_stats["state_skipped_short"] = not real_future_eligible
        video_stats["state_skipped_temporal_augmentation"] = template is not None
        return {"pixel_values": pixel_values, "grid_thw": grid,
                "question": question, "answer": answer, "index": manifest_i,
                "video_stats": video_stats}


class QACollator:
    """Builds Qwen chat-format input_ids with T*P video placeholder tokens; labels cover
    only the answer tokens. Works with the HF tokenizer (or any object exposing
    encode/eos_token_id/pad_token_id and the qwen special token ids in `cfg_ids`)."""

    def __init__(self, tokenizer, cfg_ids: dict, tokens_per_clip: int, max_len: int = 256,
                 max_answer_tokens: int = 96):
        self.tok = tokenizer
        self.ids = cfg_ids  # video_token_id / vision_start_token_id / vision_end_token_id
        self.tokens_per_clip = tokens_per_clip
        self.max_len = max_len
        self.max_answer_tokens = max_answer_tokens

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def __call__(self, batch: list[dict]) -> dict:
        seqs, labels = [], []
        answer_token_counts, answer_truncated, question_truncated = [], [], []
        for b in batch:
            pre = self._encode("<|im_start|>user\n")
            vid = [self.ids["vision_start_token_id"]] + \
                  [self.ids["video_token_id"]] * self.tokens_per_clip + \
                  [self.ids["vision_end_token_id"]]
            q_full = self._encode(b["question"] + "<|im_end|>\n<|im_start|>assistant\n")
            a_full = self._encode(b["answer"] + "<|im_end|>")

            # The old whole-sequence truncation could leave zero supervised
            # answer tokens when a native question was long.  Reserve a bounded
            # answer budget first, then truncate the question if needed.
            # Preserve the prior total-sequence ceiling: `max_len` is the text
            # budget after accounting for the visual placeholders, not in
            # addition to the user/vision prefix.
            content_budget = max(self.max_len + self.tokens_per_clip - len(pre) - len(vid), 0)
            answer_budget = min(len(a_full), self.max_answer_tokens, content_budget)
            q_budget = max(content_budget - answer_budget, 0)
            q = q_full[:q_budget]
            a = a_full[: content_budget - len(q)]
            ids = pre + vid + q + a
            lab = [-100] * (len(pre) + len(vid) + len(q)) + a
            seqs.append(ids)
            labels.append(lab)
            answer_token_counts.append(len(a))
            answer_truncated.append(int(len(a) < len(a_full)))
            question_truncated.append(int(len(q) < len(q_full)))
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        lab_t = torch.full((len(seqs), L), -100, dtype=torch.long)
        attn = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, (sequence, label_sequence) in enumerate(zip(seqs, labels)):
            input_ids[i, : len(sequence)] = torch.tensor(sequence)
            lab_t[i, : len(label_sequence)] = torch.tensor(label_sequence)
            attn[i, : len(sequence)] = 1
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "grid_thw": batch[0]["grid_thw"],
            "input_ids": input_ids, "attention_mask": attn, "labels": lab_t,
            "answer_token_count": torch.tensor(answer_token_counts, dtype=torch.float32),
            "answer_truncated": torch.tensor(answer_truncated, dtype=torch.float32),
            "question_truncated": torch.tensor(question_truncated, dtype=torch.float32),
            "indices": torch.tensor([b.get("index", -1) for b in batch], dtype=torch.long),
            "state_eligible": torch.tensor(
                [bool(b.get("video_stats", {}).get("state_eligible", False)) for b in batch],
                dtype=torch.bool,
            ),
            "video_stats": {
                key: torch.tensor(
                    [float(b.get("video_stats", {}).get(key, 0.0)) for b in batch],
                    dtype=torch.float32,
                )
                for key in (
                    "raw_frame_count", "unique_frame_count", "temporal_unit_count",
                    "duplicate_adjacent_ratio", "effective_fps", "state_skipped_short",
                    "state_skipped_temporal_augmentation",
                    "nearest_frame_substitutions", "decode_retry_count",
                )
            },
        }
