#!/usr/bin/env bash
# EXP-11 pilot: train a frozen-ViT SFT control alongside 15% mask, Orca without
# queries, and Orca with queries.  All arms share identical data, schedule, and
# step count; each launcher trains and evaluates the subset named in EXP11_ARMS.
# Stages: preflight | smoke | train | eval

set -euo pipefail

STAGE="${1:-preflight}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT="${PROJECT:-${BASE}/jepa-vlm-single-tower}"
DATA_ROOT="${DATA_ROOT:-${BASE}/jepa_data/exp10_curated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE}/results/exp11_orca_pilot}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
CLEAN_QA="${DATA_ROOT}/qa_train_clean.jsonl"
MAX_STEPS="${EXP11_MAX_STEPS:-4000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
ONLY_ARM="${ONLY_ARM:-}"

read -r -a ARMS <<< "$(printf '%s' "${EXP11_ARMS:-exp11_frozen_sft_s0 exp11_mask15_s0 exp11_orca_noquery_s0 exp11_orca_obs_s0}" | tr ',' ' ')"

info() { printf '[run-exp11] %s\n' "$*"; }
die() { printf '[run-exp11] ERROR: %s\n' "$*" >&2; exit 1; }

require_path() {
  local kind="$1" path="$2"
  [[ "$kind" == dir && -d "$path" ]] || [[ "$kind" == file && -f "$path" ]] || die "missing $kind: $path"
}

prepare_data() {
  [[ -f "$CLEAN_QA" ]] || die "clean manifest missing: $CLEAN_QA"
  info "using frozen clean manifest: $CLEAN_QA"
}

preflight() {
  local py="${JEPA_ENV:-${BASE}/jepa_env}/bin/python"
  [[ -x "$py" ]] || die "JEPA python missing: $py"
  require_path dir "$MODEL_ROOT"
  require_path file "$CLEAN_QA"
  require_path file "$MVB"
  require_path file "$TC"
  require_path file "${PROJECT}/scripts/cluster/train_multinode.sh"
  for arm in "${ARMS[@]}"; do require_path file "${PROJECT}/configs/${arm}.yaml"; done
  mkdir -p "$OUTPUT_ROOT" "$RESULTS_ROOT"
  [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
  [[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
  [[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] || die "EXP11_GRAD_ACCUM must be a positive integer"
  local world_size=$(( NPROC_PER_NODE * NNODES ))
  local effective_batch=$(( world_size * 4 * GRAD_ACCUM ))
  [[ "$effective_batch" -eq 128 ]] || die "EXP-11 effective batch must be 128; got ${effective_batch}"
  cd "$PROJECT"
  "$py" - "$MAX_STEPS" "${ARMS[@]}" <<'PY'
import sys
from jepa_vlm.config import load_config

steps = int(sys.argv[1])
for name in sys.argv[2:]:
    cfg = load_config(f"configs/{name}.yaml", [f"train.max_steps={steps}"])
    if cfg.train.sample_fps != 2.0 or cfg.train.num_frames != 16:
        raise SystemExit(f"{name}: EXP-11 arms must use 16 frames at 2 fps")
    if cfg.train.train_vision:
        raise SystemExit(f"{name}: EXP-11 freezes the visual encoder in every arm")
    print(f"preflight config: {name} steps={cfg.train.max_steps} warmup={cfg.train.warmup_steps}")
PY
  info "preflight passed: ${NPROC_PER_NODE} GPUs/node * ${NNODES} nodes * batch 4 * accum ${GRAD_ACCUM} = ${effective_batch} samples/update"
}

run_smoke() {
  cd "$PROJECT"
  for arm in "${ARMS[@]}"; do
    local out="${OUTPUT_ROOT}/exp11_smoke_${arm}"
    [[ -f "$out/step_2/state.pt" ]] && { info "$arm smoke already passed"; continue; }
    rm -rf "$out"
    info "smoke $arm"
    CONFIG="configs/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
      MASTER_ADDR=127.0.0.1 GRAD_ACCUM="$GRAD_ACCUM" \
      EXTRA_OVERRIDES="train.output_dir=${out} train.max_steps=2 train.save_every=2 train.eval_every=999999 train.log_every=1" \
      bash scripts/cluster/train_multinode.sh
    [[ -f "$out/step_2/state.pt" ]] || die "smoke failed for $arm"
  done
}

is_final_checkpoint() {
  [[ -f "$1" ]] || return 1
  "${JEPA_ENV:-${BASE}/jepa_env}/bin/python" - "$1" <<'PY'
import sys, torch
try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    raise SystemExit(1)
meta = state.get("meta", {}) if isinstance(state, dict) else {}
step = state.get("step", meta.get("step", -1)) if isinstance(state, dict) else -1
unit = meta.get("step_unit", "optimizer_update")
raise SystemExit(0 if (unit == "optimizer_update" and step >= 0) else 1)
PY
}

train_arm() {
  local arm="$1" out="${OUTPUT_ROOT}/${arm}" resume=""
  local final="${out}/step_${MAX_STEPS}"
  if is_final_checkpoint "$final/state.pt"; then
    info "$arm already complete at step_${MAX_STEPS}"
    return 0
  fi
  local latest
  latest=$(ls -d "${out}"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -n1 || true)
  if [[ -n "$latest" && -f "$latest/state.pt" ]]; then
    is_final_checkpoint "$latest/state.pt" && resume="train.resume_from=${latest}"
  fi
  cd "$PROJECT"
  info "train $arm (resume=${resume:-none})"
  CONFIG="configs/${arm}.yaml" NPROC_PER_NODE="$NPROC_PER_NODE" NNODES="$NNODES" NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" GRAD_ACCUM="$GRAD_ACCUM" \
    EXTRA_OVERRIDES="train.output_dir=${out} train.max_steps=${MAX_STEPS} ${resume}" \
    bash scripts/cluster/train_multinode.sh
  is_final_checkpoint "$final/state.pt" || die "$arm did not produce step_${MAX_STEPS}"
}

run_training() {
  [[ -n "$ONLY_ARM" ]] || die "train stage requires ONLY_ARM"
  local found=0 arm
  for arm in "${ARMS[@]}"; do [[ "$arm" == "$ONLY_ARM" ]] && found=1; done
  [[ "$found" -eq 1 ]] || die "ONLY_ARM=$ONLY_ARM is not part of EXP11_ARMS"
  train_arm "$ONLY_ARM"
}

eval_arm() {
  local arm="$1" gpu="$2" out="${OUTPUT_ROOT}/${arm}"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu" "${JEPA_ENV:-${BASE}/jepa_env}/bin/python" -m jepa_vlm.probes.mcq_eval \
      --config "$out/config.json" --ckpt "$out/step_${MAX_STEPS}" --data "$MVB" --task MVBench \
      --output "${RESULTS_ROOT}/${arm}_mvbench.json"
    CUDA_VISIBLE_DEVICES="$gpu" "${JEPA_ENV:-${BASE}/jepa_env}/bin/python" -m jepa_vlm.probes.mcq_eval \
      --config "$out/config.json" --ckpt "$out/step_${MAX_STEPS}" --data "$TC" --task Tempcompass \
      --output "${RESULTS_ROOT}/${arm}_tempcompass.json"
  ) >"${RESULTS_ROOT}/${arm}.eval.log" 2>&1
}

run_eval() {
  mkdir -p "$RESULTS_ROOT"
  local pids=() failed=0 gpu=0 arm pid
  for arm in "${ARMS[@]}"; do eval_arm "$arm" "$gpu" & pids+=("$!"); gpu=$((gpu + 1)); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  (( failed == 0 )) || die "evaluation failed; inspect ${RESULTS_ROOT}/*.eval.log"
  "${JEPA_ENV:-${BASE}/jepa_env}/bin/python" - "$RESULTS_ROOT" "${ARMS[@]}" <<'PY' | tee "${RESULTS_ROOT}/scorecard.txt"
import json, os, sys
root = sys.argv[1]
rows = []
for arm in sys.argv[2:]:
    row = {"arm": arm}
    for bench in ("mvbench", "tempcompass"):
        with open(os.path.join(root, f"{arm}_{bench}.json")) as f:
            row[bench] = json.load(f)["accuracy"]
    rows.append(row)
for r in rows:
    print(f"{r['arm']:28s} MVBench={r['mvbench']:.4f} TempCompass={r['tempcompass']:.4f}")
with open(os.path.join(root, "scorecard.json"), "w") as f:
    json.dump(rows, f, indent=2)
PY
}

case "$STAGE" in
  preflight) prepare_data; preflight ;;
  smoke) prepare_data; preflight; run_smoke ;;
  train) prepare_data; preflight; run_training ;;
  eval) run_eval ;;
  *) die "usage: $0 {preflight|smoke|train|eval}" ;;
esac
