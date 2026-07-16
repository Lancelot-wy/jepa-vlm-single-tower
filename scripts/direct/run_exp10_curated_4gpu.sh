#!/usr/bin/env bash
# Direct 4xL40S launcher for EXP-10: a source-audited caption mixture.
#
# Stages: preflight | audit | prep | smoke | train | eval | all
#
# `all` is intentionally strict: it stops before GPU work when a selected
# mounted source lacks resolvable local media or the post-filter manifest is too
# small.  The two seeds and both arms share exactly one frozen manifest.

set -euo pipefail

STAGE="${1:-all}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT="${PROJECT:-${BASE}/jepa-vlm-single-tower}"
DATA_ROOT="${DATA_ROOT:-${BASE}/jepa_data/exp10_curated}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE}/results/exp10_curated}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"

# Four complementary, user-verified processed sources. WebVid is deliberately
# absent: it adds watermarked, short-caption breadth after these sources, not
# before them.
TRAIN_SOURCES="${TRAIN_SOURCES:-llava_video vript internvid openvid1m}"
REGISTRY="${REGISTRY:-configs/data_sources_exp10.yaml}"
MIN_RAW_QA="${MIN_RAW_QA:-460000}"
MIN_TRAIN_QA="${MIN_TRAIN_QA:-440000}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

RAW_DIR="${DATA_ROOT}/raw"
RAW_QA="${RAW_DIR}/qa_train.jsonl"
CLEAN_QA="${DATA_ROOT}/qa_train_clean.jsonl"
AUDIT_REPORT="${DATA_ROOT}/source_audit.json"
PREP_FINGERPRINT="${DATA_ROOT}/prep_fingerprint.txt"

ARMS=(
  exp10_curated_sft_s0
  exp10_curated_mse_s0
  exp10_curated_sft_s1
  exp10_curated_mse_s1
)

die() { echo "[direct-exp10] ERROR: $*" >&2; exit 1; }
info() { echo "[direct-exp10] $*"; }

load_env() {
  [[ -d "$PROJECT" ]] || die "project missing: $PROJECT"
  # shellcheck disable=SC1090
  source "$PROJECT/scripts/cluster/env.cluster.sh"
  PY="${JEPA_ENV}/bin/python"
  [[ -x "$PY" ]] || die "JEPA python missing: $PY"
}

require_path() {
  local kind="$1" path="$2"
  [[ "$kind" == file && -f "$path" ]] && return
  [[ "$kind" == dir && -d "$path" ]] && return
  die "required ${kind} missing: ${path}"
}

source_args() {
  SOURCE_ARGS=()
  local source
  local sources=()
  read -r -a sources <<< "$TRAIN_SOURCES"
  (( ${#sources[@]} > 0 )) || die "TRAIN_SOURCES is empty"
  for source in "${sources[@]}"; do SOURCE_ARGS+=("$source"); done
}

preflight() {
  load_env
  require_path dir "$MODEL_ROOT"
  require_path file "$MVB"
  require_path file "$TC"
  require_path file "$PROJECT/$REGISTRY"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local gpu_count
  gpu_count="$(nvidia-smi -L | grep -c '^GPU ' || true)"
  [[ "$gpu_count" == 4 ]] || die "expected exactly 4 visible GPUs, found $gpu_count"
  mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$RESULTS_ROOT"
  local probe
  probe="$(mktemp "${DATA_ROOT}/.write_check.XXXXXX")"; rm -f "$probe"
  (
    cd "$PROJECT"
    "$PY" - <<'PY'
from jepa_vlm.config import load_config
for name in ("exp10_curated_sft_s0", "exp10_curated_sft_s1",
             "exp10_curated_mse_s0", "exp10_curated_mse_s1"):
    c = load_config(f"configs/{name}.yaml")
    print(f"{name:25s} seed={c.train.seed} lambda={c.train.lambda_reg} "
          f"mtp={c.model.mtp_enabled} min_flow={c.train.min_flow} "
          f"templates={c.train.temporal_qa_templates}")
PY
  )
  info "preflight passed: 4 GPUs; effective batch=4 GPUs * batch 4 * accum ${GRAD_ACCUM} = $((4 * 4 * GRAD_ACCUM))"
}

audit_sources() {
  load_env; source_args
  (
    cd "$PROJECT"
    "$PY" scripts/audit_data_sources.py --registry "$REGISTRY" --sources "${SOURCE_ARGS[@]}" \
      --out "$AUDIT_REPORT" --strict
  )
}

gate_manifest() {
  "$PY" - "$RAW_QA" "$CLEAN_QA" "$MIN_RAW_QA" "$MIN_TRAIN_QA" <<'PY'
import json
import sys
raw_path, clean_path, min_raw, min_clean = sys.argv[1:]
raw = sum(1 for line in open(raw_path) if line.strip())
clean = sum(1 for line in open(clean_path) if line.strip())
sources = {}
for line in open(clean_path):
    if line.strip():
        d = json.loads(line)
        sources[d.get("source_dataset", "<missing>")] = sources.get(d.get("source_dataset", "<missing>"), 0) + 1
print(f"manifest gate: raw={raw}, benchmark-ID/path-clean={clean}, sources={sources}")
if raw < int(min_raw):
    raise SystemExit(f"raw data gate failed: {raw} < {min_raw}")
if clean < int(min_clean):
    raise SystemExit(f"clean data gate failed: {clean} < {min_clean}")
if "<missing>" in sources:
    raise SystemExit("manifest provenance gate failed: source_dataset is missing")
PY
}

prep_fingerprint() {
  "$PY" - "$PROJECT/$REGISTRY" "$TRAIN_SOURCES" <<'PY'
import hashlib
import pathlib
import sys

registry, sources = sys.argv[1:]
h = hashlib.sha256()
h.update(pathlib.Path(registry).read_bytes())
h.update(b"\0")
h.update(sources.encode())
print(h.hexdigest())
PY
}

prepare_data() {
  load_env; source_args
  audit_sources
  mkdir -p "$RAW_DIR"
  local fingerprint rebuild=0
  fingerprint="$(prep_fingerprint)"
  if [[ "${FORCE_PREP:-0}" == 1 || ! -s "$RAW_QA" || ! -s "$CLEAN_QA" ]]; then
    rebuild=1
  elif [[ ! -f "$PREP_FINGERPRINT" || "$(<"$PREP_FINGERPRINT")" != "$fingerprint" ]]; then
    rebuild=1
    info "source registry or TRAIN_SOURCES changed; rebuilding stale manifest"
  fi
  if (( rebuild )); then
    rm -f "$RAW_QA" "$CLEAN_QA" "$RAW_QA.report.json"
    info "building deterministic, locally-resolved caption QA manifest from: ${TRAIN_SOURCES}"
    (
      cd "$PROJECT"
      "$PY" scripts/prepare_caption_mix.py --registry "$REGISTRY" --sources "${SOURCE_ARGS[@]}" \
        --out "$RAW_QA"
    )
  else
    info "reusing manifest with matching source fingerprint: $RAW_QA"
  fi
  [[ -s "$RAW_QA" ]] || die "caption manifest is empty"
  if (( rebuild )); then
    info "removing direct benchmark video-ID/path collisions"
    (
      cd "$PROJECT"
      "$PY" scripts/check_contamination.py --train "$RAW_QA" --bench "$MVB" "$TC" --clean-out "$CLEAN_QA"
    )
  fi
  [[ -s "$CLEAN_QA" ]] || die "benchmark-clean manifest is empty"
  gate_manifest
  printf '%s\n' "$fingerprint" > "$PREP_FINGERPRINT"
  info "data preparation passed. Framediff is deliberately not used as a filter in EXP-10."
}

run_smoke() {
  load_env
  [[ -s "$CLEAN_QA" ]] || die "run 'prep' before 'smoke'"
  local out="${OUTPUT_ROOT}/exp10_curated_smoke"
  [[ -f "$out/step_2/state.pt" ]] && { info "smoke already passed"; return; }
  mkdir -p "$out"
  (
    cd "$PROJECT"
    CONFIG=configs/exp10_curated_sft_s0.yaml NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
      MASTER_ADDR=127.0.0.1 GRAD_ACCUM="$GRAD_ACCUM" \
      EXTRA_OVERRIDES="train.output_dir=${out} train.max_steps=2 train.save_every=2 train.eval_every=999999 train.log_every=1" \
      bash scripts/cluster/train_multinode.sh
  ) 2>&1 | tee -a "$out/launcher.log"
  [[ -f "$out/step_2/state.pt" ]] || die "smoke did not produce step_2/state.pt"
}

latest_checkpoint() {
  local root="$1" candidate step best_step=-1 best=""
  for candidate in "$root"/step_*; do
    [[ -f "$candidate/state.pt" ]] || continue
    step="${candidate##*_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step > best_step )) && { best_step="$step"; best="$candidate"; }
  done
  printf '%s' "$best"
}

run_arm() {
  local arm="$1" out="${OUTPUT_ROOT}/$1" resume=""
  [[ -f "$out/step_4000/state.pt" ]] && { info "$arm already complete"; return; }
  if [[ -d "$out" && "${RESUME:-0}" == 1 ]]; then resume="$(latest_checkpoint "$out")"; fi
  if [[ -d "$out" && -z "$resume" && -f "$out/config.json" ]]; then
    die "$arm is incomplete; restart with RESUME=1"
  fi
  mkdir -p "$out"
  local overrides=""
  [[ -n "$resume" ]] && overrides="train.resume=${resume}"
  if [[ -n "$resume" ]]; then
    info "$arm: resuming from $resume"
  else
    info "$arm: starting"
  fi
  (
    cd "$PROJECT"
    CONFIG="configs/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
      GRAD_ACCUM="$GRAD_ACCUM" EXTRA_OVERRIDES="$overrides" bash scripts/cluster/train_multinode.sh
  ) 2>&1 | tee -a "$out/launcher.log"
  [[ -f "$out/step_4000/state.pt" ]] || die "$arm did not reach step_4000"
}

run_training() {
  load_env
  [[ -f "${OUTPUT_ROOT}/exp10_curated_smoke/step_2/state.pt" ]] || die "run 'smoke' before 'train'"
  local arm
  for arm in "${ARMS[@]}"; do run_arm "$arm"; done
}

eval_arm() {
  local arm="$1" gpu="$2" out="${OUTPUT_ROOT}/$1"
  (
    cd "$PROJECT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval --config "$out/config.json" \
      --ckpt "$out/step_4000" --data "$MVB" --task MVBench --output "${RESULTS_ROOT}/${arm}_mvbench.json"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval --config "$out/config.json" \
      --ckpt "$out/step_4000" --data "$TC" --task Tempcompass --output "${RESULTS_ROOT}/${arm}_tempcompass.json"
  ) >"${RESULTS_ROOT}/${arm}.eval.log" 2>&1
}

run_eval() {
  load_env; mkdir -p "$RESULTS_ROOT"
  local pids=() arm gpu=0 pid failed=0
  for arm in "${ARMS[@]}"; do eval_arm "$arm" "$gpu" & pids+=("$!"); gpu=$((gpu + 1)); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || die "evaluation failed; inspect ${RESULTS_ROOT}/*.eval.log"
  "$PY" - "$RESULTS_ROOT" "${ARMS[@]}" <<'PY' | tee "$RESULTS_ROOT/scorecard.txt"
import json, os, sys
root, *arms = sys.argv[1:]
summary = {}
for arm in arms:
    summary[arm] = {}
    for bench in ("mvbench", "tempcompass"):
        with open(os.path.join(root, f"{arm}_{bench}.json")) as f:
            d = json.load(f)
        summary[arm][bench] = {k: d[k] for k in ("acc", "correct", "total", "skipped")}
with open(os.path.join(root, "scorecard.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
}

case "$STAGE" in
  preflight) preflight ;;
  audit) preflight; audit_sources ;;
  prep) preflight; prepare_data ;;
  smoke) preflight; run_smoke ;;
  train) preflight; run_training ;;
  eval) preflight; run_eval ;;
  all) preflight; prepare_data; run_smoke; run_training; run_eval ;;
  *) die "usage: $0 {preflight|audit|prep|smoke|train|eval|all}" ;;
esac
