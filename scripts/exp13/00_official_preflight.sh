#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
SOURCE_RESULTS="${EXP12_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
ROOT="${EXP13_OFFICIAL_ROOT:-${BASE}/runs/exp13-official/preflight}"
EXPECTED_COMMIT="${EXP13_OFFICIAL_GIT_COMMIT:-}"
ATTN_IMPLEMENTATION="${EXP13_OFFICIAL_ATTN:-flash_attention_2}"

die() { echo "[exp13-official-preflight] ERROR: $*" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/.git" ]] || die "missing repository: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -z "$(git status --porcelain)" ]] || die "checkout must be clean"
HEAD="$(git rev-parse HEAD)"
[[ -z "$EXPECTED_COMMIT" || "$HEAD" == "$EXPECTED_COMMIT" ]] || die "commit mismatch"
[[ -d "$MODEL_ROOT" ]] || die "missing Qwen model: $MODEL_ROOT"
[[ -s "$MVB" ]] || die "missing MVBench JSONL: $MVB"
[[ -s "$SOURCE_RESULTS/a4_ce_k64/checkpoint-800/state.pt" ]] || die "missing A4 checkpoint"
mkdir -p "$ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
[[ "$GPU_COUNT" -eq 4 ]] || die "each evaluation Worker needs 4 GPUs; found $GPU_COUNT"

"$PY" - "$PROJECT_ROOT" "$MODEL_ROOT" "$SOURCE_RESULTS" "$MVB" \
  "$ROOT/official_budget_manifest.json" "$ATTN_IMPLEMENTATION" <<'PY'
import hashlib, importlib.util, json, os, subprocess, sys

project, model, source, mvbench, output, attention = sys.argv[1:]
from jepa_vlm.probes.mcq_eval import load_items
from jepa_vlm.probes.native_qwen_mcq_eval import (
    OFFICIAL_MAX_FRAMES, OFFICIAL_MAX_TOKENS_PER_UNIT,
    OFFICIAL_MAX_TOTAL_VIDEO_TOKENS, OFFICIAL_SAMPLE_FPS,
    build_native_inputs, load_official_frames, load_video_preprocess_config,
    official_max_pixels, official_mvbench_text,
)
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

if attention == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
    raise SystemExit("flash_attention_2 requested but flash_attn is not installed")

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

rows = load_items(mvbench, "MVBench", 0)
sample = next(
    (row for row in rows if row.get("视频") and os.path.isfile(str(row["视频"]))), None
)
if sample is None:
    examples = [str(row.get("视频")) for row in rows[:5]]
    raise SystemExit(f"no accessible real MVBench video: {examples}")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
processor = load_video_preprocess_config(model)
frames, frame_audit = load_official_frames(sample, max_frames=32, seed=0)
max_pixels = official_max_pixels(len(frames))
_, native_audit = build_native_inputs(
    tokenizer, str(sample.get("问题", "")), frames,
    float(frame_audit["timestamp_fps"]), processor,
    prompt_style="official_mvbench", max_pixels=max_pixels,
)
tokens = int(native_audit["native_video_tokens"])
units = int(native_audit["video_grid_thw"][0])
if len(frames) % 2 or len(frames) > 32:
    raise SystemExit("official frame smoke violated even/cap contract")
if tokens > OFFICIAL_MAX_TOTAL_VIDEO_TOKENS or tokens > units * OFFICIAL_MAX_TOKENS_PER_UNIT:
    raise SystemExit("official visual-token budget was exceeded")
prompt = official_mvbench_text(str(sample.get("问题", "")))
if not prompt.endswith("The best answer is:"):
    raise SystemExit("official MVBench prompt mismatch")
state = os.path.join(source, "a4_ce_k64", "checkpoint-800", "state.pt")
manifest = {
    "label": "official-budget reproduction (native-compatible HF runner; not private harness)",
    "evaluator_commit": subprocess.check_output(["git", "-C", project, "rev-parse", "HEAD"], text=True).strip(),
    "model": os.path.abspath(model),
    "checkpoint": os.path.abspath(state),
    "checkpoint_bytes": os.path.getsize(state),
    "dataset": {"path": os.path.abspath(mvbench), "sha256": digest(mvbench)},
    "public_reference": {"model": "Qwen3-VL-2B-Instruct", "MVBench_percent": 61.7},
    "budget": {"sample_fps": OFFICIAL_SAMPLE_FPS, "max_frames": OFFICIAL_MAX_FRAMES,
               "max_total_video_tokens": OFFICIAL_MAX_TOTAL_VIDEO_TOKENS,
               "max_tokens_per_unit": OFFICIAL_MAX_TOKENS_PER_UNIT},
    "attention_implementation": attention,
    "sample_preflight": {**frame_audit, **native_audit},
    "model_class": Qwen3VLForConditionalGeneration.__name__,
}
json.dump(manifest, open(output, "w"), indent=2)
print(json.dumps(manifest, indent=2))
PY

"$PY" -m jepa_vlm.probes.native_qwen_mcq_eval --help >/dev/null
echo "[exp13-official-preflight] PASS root=$ROOT attention=$ATTN_IMPLEMENTATION"
