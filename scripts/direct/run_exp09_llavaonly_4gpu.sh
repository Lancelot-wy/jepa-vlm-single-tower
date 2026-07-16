#!/usr/bin/env bash
# Direct (non-Channel-bus) execution for the LLaVA-Video-only EXP-09 derivative.
#
# Stages: preflight | prep | smoke | train | eval | all
#
# This run is intentionally named `exp09_llavaonly_*`: it does not contain an
# official NExT-QA train split and must not be reported as the original full
# EXP-09.  It preserves the original 8-GPU effective batch (4*4*8=128) on one
# 4xL40S host by setting GRAD_ACCUM=8.

set -euo pipefail

STAGE="${1:-all}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT="${PROJECT:-${BASE}/jepa-vlm-single-tower}"
DATA_ROOT="${DATA_ROOT:-${BASE}/jepa_data/exp09_llavaonly}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE}/results/exp09_llavaonly}"
LLAVA_ROOT="${LLAVA_ROOT:-/data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"

MAX_VIDEOS="${MAX_VIDEOS:-60000}"
FLOW_WORKERS="${FLOW_WORKERS:-16}"
MIN_FLOW="${MIN_FLOW:-8.42}"
MIN_RAW_QA="${MIN_RAW_QA:-100000}"
MIN_TRAIN_QA="${MIN_TRAIN_QA:-60000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

RAW_DIR="${DATA_ROOT}/raw"
RAW_QA="${RAW_DIR}/qa_train.jsonl"
CLEAN_QA="${DATA_ROOT}/qa_train_clean.jsonl"
FLOW_QA="${DATA_ROOT}/qa_train_flow.jsonl"

ARMS=(
  "exp09_llavaonly_sft_s0"
  "exp09_llavaonly_mse_s0"
  "exp09_llavaonly_sft_s1"
  "exp09_llavaonly_mse_s1"
)

die() {
  echo "[direct-exp09] ERROR: $*" >&2
  exit 1
}

info() {
  echo "[direct-exp09] $*"
}

load_env() {
  [[ -d "$PROJECT" ]] || die "project missing: $PROJECT"
  # shellcheck disable=SC1090
  source "$PROJECT/scripts/cluster/env.cluster.sh"
  PY="${JEPA_ENV}/bin/python"
  [[ -x "$PY" ]] || die "JEPA python missing: $PY"
}

require_path() {
  local kind="$1" path="$2"
  if [[ "$kind" == "file" ]]; then
    [[ -f "$path" ]] || die "required file missing: $path"
  else
    [[ -d "$path" ]] || die "required directory missing: $path"
  fi
}

preflight() {
  load_env
  require_path dir "$LLAVA_ROOT"
  require_path dir "$MODEL_ROOT"
  require_path file "$MVB"
  require_path file "$TC"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local gpu_count
  gpu_count="$(nvidia-smi -L | grep -c '^GPU ' || true)"
  [[ "$gpu_count" == "4" ]] || die "expected exactly 4 visible GPUs, found $gpu_count"
  mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$RESULTS_ROOT"
  local write_probe
  write_probe="$(mktemp "${DATA_ROOT}/.write_check.XXXXXX")"
  rm -f "$write_probe"

  (
    cd "$PROJECT"
    "$PY" - <<'PY'
from jepa_vlm.config import load_config

for name in (
    "exp09_llavaonly_sft_s0", "exp09_llavaonly_sft_s1",
    "exp09_llavaonly_mse_s0", "exp09_llavaonly_mse_s1",
):
    c = load_config(f"configs/{name}.yaml")
    print(f"{name:27s} seed={c.train.seed} lambda={c.train.lambda_reg} "
          f"mtp={c.model.mtp_enabled} reg={c.model.reg_enabled} "
          f"min_flow={c.train.min_flow} templates={c.train.temporal_qa_templates}")
PY
  )
  info "preflight passed: 4 GPUs; direct world=4; GRAD_ACCUM=${GRAD_ACCUM}; effective batch=4*${GRAD_ACCUM}*4=$((4 * GRAD_ACCUM * 4))"
}

gate_manifest() {
  "$PY" - "$RAW_QA" "$CLEAN_QA" "$FLOW_QA" "$MIN_FLOW" "$MIN_RAW_QA" "$MIN_TRAIN_QA" <<'PY'
import json
import math
import sys

raw_path, clean_path, flow_path, min_flow, min_raw, min_train = sys.argv[1:]
min_flow = float(min_flow)
min_raw = int(min_raw)
min_train = int(min_train)

raw = sum(1 for line in open(raw_path) if line.strip())
clean = sum(1 for line in open(clean_path) if line.strip())
total = valid = kept = bad = 0
for line in open(flow_path):
    if not line.strip():
        continue
    total += 1
    d = json.loads(line)
    value = d.get("flow")
    if isinstance(value, (int, float)) and math.isfinite(value):
        valid += 1
        kept += value >= min_flow
    else:
        bad += 1

print(f"manifest gate: raw_qa={raw}, contamination_clean={clean}, flow_rows={total}, valid_flow={valid}, "
      f"decode_failed={bad}, retained_at_{min_flow:g}={kept}")
if raw < min_raw:
    raise SystemExit(f"raw QA gate failed: {raw} < {min_raw}")
if total != clean:
    raise SystemExit(f"row-count mismatch: flow {total} != contamination-clean {clean}")
if kept < min_train:
    raise SystemExit(f"retained QA gate failed: {kept} < {min_train}")
PY
}

prepare_data() {
  load_env
  mkdir -p "$RAW_DIR"
  if [[ ! -s "$RAW_QA" ]]; then
    info "building a uniform all-source LLaVA reservoir sample (max_videos=${MAX_VIDEOS})"
    (
      cd "$PROJECT"
      "$PY" scripts/prepare_llava_video.py \
        --root "$LLAVA_ROOT" \
        --all-subsets \
        --exclude-patterns activitynet perceptiontest perception_test charades nextqa next-qa \
        --out-dir "$RAW_DIR" \
        --max-videos "$MAX_VIDEOS" \
        --qa --qa-per-video 2
    )
  else
    info "reusing existing raw QA manifest: $RAW_QA"
  fi
  [[ -s "$RAW_QA" ]] || die "LLaVA QA extraction produced no manifest"

  if [[ ! -s "$CLEAN_QA" ]]; then
    info "running benchmark contamination filter"
    (
      cd "$PROJECT"
      "$PY" scripts/check_contamination.py \
        --train "$RAW_QA" --bench "$MVB" "$TC" --clean-out "$CLEAN_QA"
    )
  else
    info "reusing existing contamination-clean manifest: $CLEAN_QA"
  fi
  [[ -s "$CLEAN_QA" ]] || die "contamination filter produced an empty manifest"

  info "computing per-video flow once and expanding it to every QA row (workers=${FLOW_WORKERS})"
  (
    cd "$PROJECT"
    "$PY" scripts/compute_flow.py \
      --manifest "$CLEAN_QA" --out "$FLOW_QA" --method framediff \
      --workers "$FLOW_WORKERS" --resume
  )
  [[ -s "$FLOW_QA" ]] || die "flow output missing: $FLOW_QA"
  gate_manifest
  info "data preparation passed all gates"
}

run_smoke() {
  load_env
  [[ -s "$FLOW_QA" ]] || die "run 'prep' before 'smoke'"
  local smoke_out="${OUTPUT_ROOT}/exp09_llavaonly_smoke"
  if [[ -f "$smoke_out/step_2/state.pt" ]]; then
    info "smoke checkpoint already exists: $smoke_out/step_2"
    return
  fi
  mkdir -p "$smoke_out"
  info "starting a 2-step, 4-GPU smoke test"
  (
    cd "$PROJECT"
    CONFIG="configs/exp09_llavaonly_sft_s0.yaml" \
      NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
      GRAD_ACCUM="$GRAD_ACCUM" \
      EXTRA_OVERRIDES="train.min_flow=${MIN_FLOW} train.output_dir=${smoke_out} train.max_steps=2 train.save_every=2 train.eval_every=999999 train.log_every=1" \
      bash scripts/cluster/train_multinode.sh
  ) 2>&1 | tee -a "$smoke_out/launcher.log"
  [[ -f "$smoke_out/step_2/state.pt" ]] || die "smoke did not produce step_2/state.pt"
  info "smoke passed"
}

latest_checkpoint() {
  local root="$1" best_step=-1 best=""
  local candidate step
  for candidate in "$root"/step_*; do
    [[ -f "$candidate/state.pt" ]] || continue
    step="${candidate##*_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step > best_step )); then
      best_step="$step"
      best="$candidate"
    fi
  done
  printf '%s' "$best"
}

run_arm() {
  local arm="$1"
  local out="${OUTPUT_ROOT}/${arm}"
  if [[ -f "$out/step_4000/state.pt" ]]; then
    info "${arm}: already complete"
    return
  fi
  local resume=""
  if [[ -d "$out" && "${RESUME:-0}" == "1" ]]; then
    resume="$(latest_checkpoint "$out")"
  fi
  if [[ -d "$out" && -z "$resume" && -f "$out/config.json" ]]; then
    die "${arm} has an incomplete run. Restart with RESUME=1 to resume its newest checkpoint."
  fi
  mkdir -p "$out"
  local overrides="train.min_flow=${MIN_FLOW}"
  if [[ -n "$resume" ]]; then
    info "${arm}: resuming from ${resume}"
    overrides+=" train.resume=${resume}"
  else
    info "${arm}: starting"
  fi
  (
    cd "$PROJECT"
    CONFIG="configs/${arm}.yaml" \
      NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
      GRAD_ACCUM="$GRAD_ACCUM" EXTRA_OVERRIDES="$overrides" \
      bash scripts/cluster/train_multinode.sh
  ) 2>&1 | tee -a "$out/launcher.log"
  [[ -f "$out/step_4000/state.pt" ]] || die "${arm}: no final step_4000 checkpoint"
}

run_training() {
  load_env
  [[ -f "${OUTPUT_ROOT}/exp09_llavaonly_smoke/step_2/state.pt" ]] || die "run 'smoke' before 'train'"
  for arm in "${ARMS[@]}"; do
    run_arm "$arm"
  done
  info "all four training arms completed"
}

eval_arm() {
  local arm="$1" gpu="$2" out="${OUTPUT_ROOT}/${arm}"
  local log="${RESULTS_ROOT}/${arm}.eval.log"
  [[ -f "$out/step_4000/state.pt" ]] || die "missing checkpoint for ${arm}"
  (
    cd "$PROJECT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval \
      --config "$out/config.json" --ckpt "$out/step_4000" \
      --data "$MVB" --task MVBench --output "${RESULTS_ROOT}/${arm}_mvbench.json"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval \
      --config "$out/config.json" --ckpt "$out/step_4000" \
      --data "$TC" --task Tempcompass --output "${RESULTS_ROOT}/${arm}_tempcompass.json"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.temporal_qa_eval \
      --config "$out/config.json" --ckpt "$out/step_4000" \
      --manifest "${RAW_DIR}/val.jsonl" --max-clips 500 \
      | tee "${RESULTS_ROOT}/${arm}_temporalqa.txt"
  ) >"$log" 2>&1
}

write_scorecard() {
  "$PY" - "$RESULTS_ROOT" "${ARMS[@]}" <<'PY'
import json
import os
import sys

root, *arms = sys.argv[1:]
summary = {}
for arm in arms:
    summary[arm] = {}
    for bench in ("mvbench", "tempcompass"):
        path = os.path.join(root, f"{arm}_{bench}.json")
        with open(path) as f:
            d = json.load(f)
        summary[arm][bench] = {k: d[k] for k in ("acc", "correct", "total", "skipped")}

with open(os.path.join(root, "scorecard.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
}

run_eval() {
  load_env
  mkdir -p "$RESULTS_ROOT"
  local pids=() arm gpu=0 pid failed=0
  for arm in "${ARMS[@]}"; do
    eval_arm "$arm" "$gpu" &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || die "one or more evaluation workers failed; inspect ${RESULTS_ROOT}/*.eval.log"
  write_scorecard | tee "${RESULTS_ROOT}/scorecard.txt"
  info "evaluation complete: ${RESULTS_ROOT}/scorecard.json"
}

case "$STAGE" in
  preflight) preflight ;;
  prep) preflight; prepare_data ;;
  smoke) preflight; run_smoke ;;
  train) preflight; run_training ;;
  eval) preflight; run_eval ;;
  all) preflight; prepare_data; run_smoke; run_training; run_eval ;;
  *) die "usage: $0 {preflight|prep|smoke|train|eval|all}" ;;
esac
