"""Export/apply EXP-12 trainable tensors to a native Qwen3-VL model.

Training checkpoints also contain optimizer/scheduler state.  Native generation is
usually sharded across GPUs, so loading the full training checkpoint in every shard
would multiply host-memory use.  ``export_native_overlay`` reads it once and writes
only tensors that belong to the Hugging Face Qwen model.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping

import torch


def map_wrapper_state_to_native(
    wrapper_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map ``JepaQwen3VL`` backbone keys to HF top-level Qwen3-VL keys."""
    mapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    prefixes = {
        "language_model.": "model.language_model.",
        "visual.": "model.visual.",
        "lm_head.": "lm_head.",
    }
    for key, tensor in wrapper_state.items():
        for source, target in prefixes.items():
            if key.startswith(source):
                mapped[target + key[len(source):]] = tensor
                break
        else:
            ignored.append(key)
    if not any(key.startswith("model.language_model.") for key in mapped):
        raise ValueError("checkpoint contains no trainable language-model tensors")
    return mapped, ignored


def export_native_overlay(checkpoint: str, output: str) -> dict:
    """Write a compact native-key overlay from one ``state.pt`` checkpoint."""
    state_path = checkpoint
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(state_path)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if "model" not in payload or not isinstance(payload["model"], Mapping):
        raise ValueError(f"not a JEPA training checkpoint: {state_path}")
    mapped, ignored = map_wrapper_state_to_native(payload["model"])
    metadata = {
        "schema_version": 1,
        "source_checkpoint": os.path.abspath(state_path),
        "source_step": payload.get("step"),
        "mapped_tensor_count": len(mapped),
        "ignored_tensor_count": len(ignored),
        "ignored_prefixes": sorted({key.split(".", 1)[0] for key in ignored}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    temporary = output + ".tmp"
    torch.save({"model": mapped, "metadata": metadata}, temporary)
    os.replace(temporary, output)
    with open(output + ".json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def apply_native_overlay(model, overlay_path: str) -> dict:
    """Load an exported overlay into ``Qwen3VLForConditionalGeneration``."""
    payload = torch.load(overlay_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"invalid native overlay: {overlay_path}")
    native_state = model.state_dict()
    unexpected = sorted(set(state) - set(native_state))
    mismatched = [
        key for key in state.keys() & native_state.keys()
        if tuple(state[key].shape) != tuple(native_state[key].shape)
    ]
    if unexpected or mismatched:
        raise ValueError(
            f"overlay/model mismatch: unexpected={unexpected[:5]} mismatched={mismatched[:5]}"
        )
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected overlay keys: {result.unexpected_keys[:5]}")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload, dict) else {}
    metadata.update({
        "overlay": os.path.abspath(overlay_path),
        "loaded_tensor_count": len(state),
    })
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or state.pt")
    parser.add_argument("--output", required=True, help="output .pt containing native-key tensors only")
    args = parser.parse_args()
    print(json.dumps(export_native_overlay(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
