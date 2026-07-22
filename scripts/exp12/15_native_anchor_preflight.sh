#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
SOURCE_RESULTS="${EXP12_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ROOT="${EXP12_NATIVE_ANCHOR_ROOT:-${BASE}/runs/exp13/native-anchor}"

die() { echo "[native-anchor-preflight] ERROR: $*" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/.git" ]] || die "missing repository: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
if [[ -f scripts/cluster/env.cluster.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/env.cluster.sh
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"
PY="${PY:-$(command -v python3 || true)}"
[[ -x "$PY" ]] || die "Python unavailable"
[[ -d "$MODEL_ROOT" ]] || die "missing Qwen model: $MODEL_ROOT"
[[ -s "$MVB" && -s "$TC" ]] || die "missing MVBench/TempCompass JSONL"
[[ -s "$SOURCE_RESULTS/a4_ce_k64/checkpoint-800/state.pt" ]] || die "missing a4 K64 checkpoint"
[[ -s "$SOURCE_RESULTS/a4_ce_k64/config.json" ]] || die "missing a4 config.json"
mkdir -p "$ROOT"

"$PY" - "$PROJECT_ROOT" "$MODEL_ROOT" "$SOURCE_RESULTS" "$MVB" "$TC" "$ROOT/protocol_manifest.json" <<'PY'
import hashlib, json, os, subprocess, sys

project, model, source, mvbench, tempcompass, output = sys.argv[1:]
from jepa_vlm.config import load_config, resolved_visual_tokens
from jepa_vlm.probes.mcq_eval import load_items
from jepa_vlm.probes.native_qwen_mcq_eval import (
    build_native_inputs, load_native_frames, load_video_preprocess_config,
)
from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration
from PIL import Image

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

ks = []
for arm in ("a0_ce_k4", "a2_ce_k16", "a4_ce_k64"):
    cfg = load_config(os.path.join(project, "configs/orca_token_sweep", arm + ".yaml"))
    ks.append(resolved_visual_tokens(cfg))
if ks != [4, 16, 64]:
    raise SystemExit(f"unexpected K controls: {ks}")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
configuration = AutoConfig.from_pretrained(model, local_files_only=True)
video_processor = load_video_preprocess_config(model)
native_samples = {}
for path, task in ((mvbench, "MVBench"), (tempcompass, "Tempcompass")):
    rows = load_items(path, task, 2)
    if not rows:
        raise SystemExit(f"no {task} rows in {path}")
    if not any((row.get("meta") or {}).get("images_info") or row.get("视频") for row in rows):
        raise SystemExit(f"{task} sample has neither images_info nor video")
    sample = rows[0]
    images = (sample.get("meta") or {}).get("images_info") or []
    if images:
        image_path = images[0].get("image")
        if not image_path or not os.path.isfile(image_path):
            raise SystemExit(f"{task} sample image is inaccessible: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
    else:
        video_path = sample.get("视频")
        if not video_path or not os.path.isfile(video_path):
            raise SystemExit(f"{task} sample video is inaccessible: {video_path}")
    frames, frame_audit = load_native_frames(sample, 32, 4.0, 0)
    _, native_audit = build_native_inputs(
        tokenizer, str(sample.get("问题", "")), frames,
        float(frame_audit["timestamp_fps"]), video_processor,
    )
    native_samples[task] = {**frame_audit, **native_audit}
state = os.path.join(source, "a4_ce_k64", "checkpoint-800", "state.pt")
manifest = {
    "evaluator_commit": subprocess.check_output(["git", "-C", project, "rev-parse", "HEAD"], text=True).strip(),
    "training_commit": open(os.path.join(source, "a4_ce_k64", "git_commit.txt")).read().strip(),
    "model": os.path.abspath(model),
    "source_results": os.path.abspath(source),
    "checkpoint": os.path.abspath(state),
    "checkpoint_bytes": os.path.getsize(state),
    "datasets": {
        "MVBench": {"path": os.path.abspath(mvbench), "sha256": digest(mvbench)},
        "Tempcompass": {"path": os.path.abspath(tempcompass), "sha256": digest(tempcompass)},
    },
    "controls": {"K": ks, "raw_frames": 32, "temporal_units": 16},
    "native_sample_preflight": native_samples,
    "native_classes": {
        "tokenizer": tokenizer.__class__.__name__,
        "processor": "torchvision_free_qwen3vl_compat_v1",
        "video_preprocessor_config": video_processor["_path"],
        "model": Qwen3VLForConditionalGeneration.__name__,
        "model_type": configuration.model_type,
    },
}
with open(output, "w") as handle:
    json.dump(manifest, handle, indent=2)
print(json.dumps(manifest, indent=2))
PY

"$PY" -m jepa_vlm.probes.mcq_eval --help >/dev/null
"$PY" -m jepa_vlm.probes.native_qwen_mcq_eval --help >/dev/null
echo "[native-anchor-preflight] PASS root=$ROOT"
