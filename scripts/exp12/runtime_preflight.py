#!/usr/bin/env python3
"""Runtime checks that require the server Python/model environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import subprocess

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


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mvbench", required=True)
    parser.add_argument("--tempcompass", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0)
    args = parser.parse_args()

    import torch
    import transformers

    project = pathlib.Path(args.project).resolve()
    os.chdir(project)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.expected_commit and commit != args.expected_commit:
        raise SystemExit(f"commit mismatch: {commit} != {args.expected_commit}")
    for path in (args.manifest, args.model, args.mvbench, args.tempcompass):
        if not os.path.exists(path):
            raise SystemExit(f"missing required path: {path}")

    configs = []
    for arm in ARMS:
        cfg = load_config(f"configs/orca_token_sweep/{arm}.yaml")
        if pathlib.Path(cfg.model.pretrained) != pathlib.Path(args.model):
            raise SystemExit(f"{arm}: model path differs from preflight model")
        if pathlib.Path(cfg.train.text_manifest) != pathlib.Path(args.manifest):
            raise SystemExit(f"{arm}: manifest differs from frozen manifest")
        if resolved_raw_num_frames(cfg) != 32 or resolved_temporal_units(cfg) != 16:
            raise SystemExit(f"{arm}: temporal contract mismatch")
        configs.append(cfg)
    if [resolved_visual_tokens(cfg) for cfg in configs] != [4, 4, 16, 16, 64, 64]:
        raise SystemExit("K sweep mismatch")

    # Decode a deterministic reservoir to catch inaccessible mounts/corrupt media
    # without making one short random clip fail an otherwise healthy corpus.
    from jepa_vlm.data.video_io import decode_frames

    rng = random.Random(120012)
    candidates = []
    row_count = 0
    missing_fields = 0
    with open(args.manifest) as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            missing_fields += int(any(not row.get(key) for key in (
                "video", "question", "answer", "source_dataset"
            )))
            if len(candidates) < 8:
                candidates.append(row)
            else:
                replacement = rng.randrange(row_count)
                if replacement < len(candidates):
                    candidates[replacement] = row
    if not candidates:
        raise SystemExit("empty training manifest")
    if missing_fields:
        raise SystemExit(f"manifest has {missing_fields} rows missing required QA/provenance fields")
    decode_attempts = []
    diagnostics = None
    for row in candidates:
        try:
            frames, candidate_diagnostics = decode_frames(
                row["video"], 32, 4.0, "fps_or_uniform",
                start=row.get("start"), end=row.get("end"), random_offset=False,
                return_metadata=True, temporal_patch_size=2, state_horizon_units=2,
            )
            if frames.shape[0] != 32:
                raise RuntimeError("decode did not return 32 frames")
            decode_attempts.append({
                "video": row["video"], "status": "ok", **candidate_diagnostics,
            })
            if candidate_diagnostics["state_eligible"]:
                diagnostics = candidate_diagnostics
                break
        except Exception as exc:
            decode_attempts.append({
                "video": row.get("video", ""), "status": "error", "error": str(exc),
            })
    if diagnostics is None:
        raise SystemExit(
            "no sampled manifest row decoded into 32 unique real frames and 16 valid units: "
            + json.dumps(decode_attempts, ensure_ascii=False)
        )

    parameter = None
    if args.load_model:
        from jepa_vlm.modeling.model import build_model
        from jepa_vlm.train import make_optimizer, parameter_audit

        model = build_model(configs[1])
        optimizer = make_optimizer(model, configs[1])
        parameter = parameter_audit(model, optimizer, configs[1])
        if parameter["visual_parameters_in_optimizer"]:
            raise SystemExit("visual parameters entered optimizer")
        model.assert_exp12_frozen_visual()
        del optimizer, model

    world_size = args.world_size or max(torch.cuda.device_count(), 1)
    grad_accum = args.grad_accum or configs[0].train.grad_accum
    effective_batch = configs[0].train.batch_size * world_size * grad_accum
    if effective_batch != 32:
        raise SystemExit(f"effective batch must be 32, got {effective_batch}")
    result = {
        "commit": commit,
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "manifest_rows": row_count,
        "model": os.path.abspath(args.model),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "decode": diagnostics,
        "decode_attempts": decode_attempts,
        "framework": {
            "trainer": "custom torch + Accelerate loop",
            "llama_factory_installed": importlib.util.find_spec("llamafactory") is not None,
        },
        "requested_world_size": world_size,
        "requested_grad_accum": grad_accum,
        "effective_batch_size": effective_batch,
        "configs": [
            {
                "arm": arm,
                "K": resolved_visual_tokens(cfg),
                "mode": cfg.model.state_predictor_mode,
                "effective_batch": cfg.train.batch_size * grad_accum * world_size,
            }
            for arm, cfg in zip(ARMS, configs)
        ],
        "parameter_audit": parameter,
    }
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "preflight.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
