#!/usr/bin/env bash
# EXP-11 overnight pilot: frozen-SFT control, 15% mask, minimal Orca observation loss.
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
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${EXP11_MAX_STEPS:-1000}"
SAVE_EVERY="${EXP11_SAVE_EVERY:-250}"

ARMS=(
  exp11_frozen_sft_s0
  exp11_mask15_s0
  exp11_orca_obs_s0
)

die() { echo "[direct-exp11] ERROR: $*" >&2; exit 1; }
info() { echo "[direct-exp11] $*"; }

load_env() {
  [[ -d "$PROJECT" ]] || die "project missing: $PROJECT"
  # shellcheck disable=SC1090
  source "$PROJECT/scripts/cluster/env.cluster.sh"
  PY="${JEPA_ENV}/bin/python"
  [[ -x "$PY" ]] || die "JEPA python missing: $PY"
}

preflight() {
  load_env
  [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || die "EXP11_MAX_STEPS must be positive"
  [[ "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || die "EXP11_SAVE_EVERY must be positive"
  [[ -d "$MODEL_ROOT" ]] || die "model missing: $MODEL_ROOT"
  [[ -s "$CLEAN_QA" ]] || die "clean EXP-10 manifest missing: $CLEAN_QA (run EXP-10 prep first)"
  [[ -f "$MVB" && -f "$TC" ]] || die "MVBench/TempCompass manifest missing"
  local gpu_count
  gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  [[ "$gpu_count" == 4 ]] || die "expected exactly 4 visible GPUs, found $gpu_count"
  mkdir -p "$OUTPUT_ROOT" "$RESULTS_ROOT"
  (
    cd "$PROJECT"
    "$PY" - "$MAX_STEPS" <<'PY'
import sys
from jepa_vlm.config import load_config

expected_steps = int(sys.argv[1])
names = ("exp11_frozen_sft_s0", "exp11_mask15_s0", "exp11_orca_obs_s0")
for name in names:
    c = load_config(f"configs/{name}.yaml", [f"train.max_steps={expected_steps}"])
    if c.train.sample_fps != 2.0 or c.train.num_frames != 16:
        raise SystemExit(f"{name}: expected the paired 2-fps/16-frame regime")
    if c.train.train_vision:
        raise SystemExit(f"{name}: EXP-11 pilot requires the paired frozen ViT control")
    print(
        f"{name:26s} mask={c.model.mask_variant}:{c.model.mask_ratio} "
        f"dual={c.model.dual_view} orca={c.model.orca_enabled} "
        f"gap={c.model.orca_target_gap} lambda={c.train.lambda_reg} steps={c.train.max_steps}"
    )
PY
  )
  info "preflight passed: ${NPROC_PER_NODE} GPU/node * ${NNODES} nodes * batch 4 * accum ${GRAD_ACCUM} = $((NPROC_PER_NODE * NNODES * 4 * GRAD_ACCUM)) samples/update; checkpoints every ${SAVE_EVERY} updates"
}

run_smoke() {
  load_env
  local arm out
  for arm in "${ARMS[@]}"; do
    out="${OUTPUT_ROOT}/smoke_${arm}"
    [[ -f "$out/step_2/state.pt" ]] && { info "$arm smoke already passed"; continue; }
    mkdir -p "$out"
    (
      cd "$PROJECT"
      CONFIG="configs/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
        MASTER_ADDR=127.0.0.1 GRAD_ACCUM=1 \
        EXTRA_OVERRIDES="train.output_dir=${out} train.max_steps=2 train.warmup_steps=1 train.save_every=2 train.eval_every=999999 train.log_every=1 train.num_workers=1" \
        bash scripts/cluster/train_multinode.sh
    ) 2>&1 | tee -a "$out/launcher.log"
    [[ -f "$out/step_2/state.pt" ]] || die "$arm smoke did not produce step_2/state.pt"
    "$PY" - "$arm" "$out/log.jsonl" <<'PY'
import json, math, sys
arm, path = sys.argv[1:]
rows = [json.loads(line) for line in open(path) if line.strip()]
row = next((r for r in reversed(rows) if r.get("step") == 2 and "lr" in r), None)
if row is None:
    raise SystemExit(f"{arm}: no step-2 train metrics")
for key in ("loss", "ce_loss"):
    if key not in row or not math.isfinite(row[key]):
        raise SystemExit(f"{arm}: invalid {key}: {row.get(key)}")
if arm == "exp11_mask15_s0":
    frac = row.get("mask_fraction")
    if "reg_loss" not in row or frac is None or not 0.10 <= frac <= 0.20:
        raise SystemExit(f"{arm}: invalid mask smoke metrics: {row}")
if arm == "exp11_orca_obs_s0":
    required = ("orca_loss", "orca_persistence_mse", "orca_persistence_ratio",
                "orca_target_std", "orca_pred_std", "orca_frame_encoding_mse")
    if any(k not in row or not math.isfinite(row[k]) for k in required):
        raise SystemExit(f"{arm}: missing/non-finite Orca metrics: {row}")
    if row["orca_persistence_mse"] <= 0 or row["orca_target_std"] <= 0.05:
        raise SystemExit(f"{arm}: degenerate frozen targets: {row}")
    if row["orca_frame_encoding_mse"] > 1e-4:
        raise SystemExit(
            f"{arm}: native and explicit per-frame encodings differ: "
            f"{row['orca_frame_encoding_mse']}"
        )
print(f"{arm}: smoke metric gate passed")
PY
  done
}

is_update_checkpoint() {
  "$PY" - "$1" <<'PY'
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if s.get("step_unit") == "optimizer_update" else 1)
PY
}

latest_checkpoint() {
  local root="$1" candidate step best_step=-1 best=""
  for candidate in "$root"/step_*; do
    [[ -f "$candidate/state.pt" ]] || continue
    step="${candidate##*_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    is_update_checkpoint "$candidate/state.pt" >/dev/null 2>&1 || continue
    (( step > best_step )) && { best_step="$step"; best="$candidate"; }
  done
  printf '%s' "$best"
}

run_arm() {
  local arm="$1" out="${OUTPUT_ROOT}/$1" resume=""
  local launch_nproc="${TRAIN_NPROC_PER_NODE:-4}"
  local launch_nnodes="${TRAIN_NNODES:-1}"
  local launch_node_rank="${TRAIN_NODE_RANK:-0}"
  local launch_master="${TRAIN_MASTER_ADDR:-127.0.0.1}"
  local final="${out}/step_${MAX_STEPS}/state.pt"
  if [[ -f "$final" ]]; then
    is_update_checkpoint "$final" || die "$arm final checkpoint uses legacy accounting"
    info "$arm already complete"
    return
  fi
  if [[ -d "$out" && "${RESUME:-0}" == 1 ]]; then resume="$(latest_checkpoint "$out")"; fi
  if [[ -d "$out" && -z "$resume" && -f "$out/config.json" ]]; then
    die "$arm is incomplete; resubmit with --resume and the same run id"
  fi
  mkdir -p "$out"
  local overrides="train.output_dir=${out} train.max_steps=${MAX_STEPS} train.warmup_steps=$((MAX_STEPS / 10)) train.save_every=${SAVE_EVERY}"
  [[ -n "$resume" ]] && overrides+=" train.resume=${resume}"
  [[ -n "${TRAIN_EXTRA_OVERRIDES:-}" ]] && overrides+=" ${TRAIN_EXTRA_OVERRIDES}"
  if [[ -n "$resume" ]]; then info "$arm: resuming from $resume"; else info "$arm: starting fresh"; fi
  (
    cd "$PROJECT"
    CONFIG="configs/${arm}.yaml" NPROC_PER_NODE="$launch_nproc" NNODES="$launch_nnodes" \
      NODE_RANK="$launch_node_rank" MASTER_ADDR="$launch_master" GRAD_ACCUM="$GRAD_ACCUM" \
      EXTRA_OVERRIDES="$overrides" bash scripts/cluster/train_multinode.sh
  ) 2>&1 | tee -a "${out}/launcher_rank${launch_node_rank}.log"
  [[ -f "$final" ]] || die "$arm did not reach step_${MAX_STEPS}"
}

run_training() {
  load_env
  local arm
  for arm in "${ARMS[@]}"; do
    [[ -f "${OUTPUT_ROOT}/smoke_${arm}/step_2/state.pt" ]] || die "missing smoke checkpoint for $arm"
  done
  if [[ -n "${ONLY_ARM:-}" ]]; then
    for arm in "${ARMS[@]}"; do
      [[ "$arm" == "$ONLY_ARM" ]] && { run_arm "$arm"; return; }
    done
    die "unknown ONLY_ARM=${ONLY_ARM}"
  fi
  for arm in "${ARMS[@]}"; do run_arm "$arm"; done
}

eval_arm() {
  local arm="$1" gpu="$2" out="${OUTPUT_ROOT}/$1" ckpt="${OUTPUT_ROOT}/$1/step_${MAX_STEPS}"
  (
    cd "$PROJECT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval --config "$out/config.json" \
      --ckpt "$ckpt" --data "$MVB" --task MVBench --output "${RESULTS_ROOT}/${arm}_mvbench.json"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval --config "$out/config.json" \
      --ckpt "$ckpt" --data "$TC" --task Tempcompass --output "${RESULTS_ROOT}/${arm}_tempcompass.json"
  ) >"${RESULTS_ROOT}/${arm}.eval.log" 2>&1
}

run_eval() {
  load_env; mkdir -p "$RESULTS_ROOT"
  local pids=() arm gpu=0 pid failed=0
  for arm in "${ARMS[@]}"; do eval_arm "$arm" "$gpu" & pids+=("$!"); gpu=$((gpu + 1)); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || die "evaluation failed; inspect ${RESULTS_ROOT}/*.eval.log"
  "$PY" - "$RESULTS_ROOT" "$OUTPUT_ROOT" "$MAX_STEPS" "${ARMS[@]}" <<'PY' | tee "$RESULTS_ROOT/scorecard.txt"
import json, os, sys
root, outputs, max_steps, *arms = sys.argv[1:]
summary = {"max_steps": int(max_steps), "arms": {}}
for arm in arms:
    row = {}
    for bench in ("mvbench", "tempcompass"):
        with open(os.path.join(root, f"{arm}_{bench}.json")) as f:
            d = json.load(f)
        row[bench] = {k: d[k] for k in ("acc", "correct", "total", "skipped")}
    log_path = os.path.join(outputs, arm, "log.jsonl")
    last = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            rec = json.loads(line)
            if rec.get("step") == int(max_steps) and "lr" in rec:
                last = rec
    row["final_train"] = {k: last[k] for k in (
        "loss", "ce_loss", "reg_loss", "orca_loss", "orca_weighted_loss",
        "orca_persistence_mse", "orca_persistence_ratio", "orca_gain_vs_persistence",
        "orca_target_std", "orca_pred_std", "orca_frame_encoding_mse",
        "target_std", "copy_mse", "mask_fraction"
    ) if k in last}
    summary["arms"][arm] = row
control = summary["arms"]["exp11_frozen_sft_s0"]
for arm, row in summary["arms"].items():
    row["delta_vs_control_pp"] = {
        bench: 100.0 * (row[bench]["acc"] - control[bench]["acc"])
        for bench in ("mvbench", "tempcompass")
    }
with open(os.path.join(root, "scorecard.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
}

case "$STAGE" in
  preflight) preflight ;;
  smoke) preflight; run_smoke ;;
  train) preflight; run_training ;;
  eval) preflight; run_eval ;;
  *) die "usage: $0 {preflight|smoke|train|eval}" ;;
esac
