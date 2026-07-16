#!/usr/bin/env bash
# High-throughput vivolm entrypoint for EXP-10.
#
# vtraining gives every Worker the same TF_CONFIG.  This script partitions the
# ordered Worker list into four independent DDP groups instead of accidentally
# creating one 128-GPU run: group 0 -> sft_s0, group 1 -> mse_s0, group 2 ->
# sft_s1, group 3 -> mse_s1.  Each group preserves effective batch 128.

set -Eeuo pipefail

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
RUN_ID="${EXP10_RUN_ID:-}"
ATTEMPT_ID="${EXP10_ATTEMPT_ID:-}"
RESUME="${EXP10_RESUME:-0}"
NODES_PER_ARM="${EXP10_NODES_PER_ARM:-8}"
GRAD_ACCUM="${EXP10_GRAD_ACCUM:-1}"
NUM_WORKERS="${EXP10_NUM_WORKERS:-2}"
SAVE_EVERY="${EXP10_SAVE_EVERY:-250}"
WAIT_TIMEOUT_SEC="${EXP10_WAIT_TIMEOUT_SEC:-86400}"
ARMS=(
  exp10_curated_sft_s0
  exp10_curated_mse_s0
  exp10_curated_sft_s1
  exp10_curated_mse_s1
)
COORD_DIR=""

record_failure() {
  local code="${1:-1}" message="${2:-unspecified failure}"
  [[ -n "$COORD_DIR" ]] || return 0
  mkdir -p "$COORD_DIR" || true
  printf 'code=%s host=%s rank=%s group=%s message=%s time=%s\n' \
    "$code" "$(hostname)" "${PLATFORM_RANK:-unknown}" "${GROUP_ID:-unknown}" \
    "$message" "$(date -Is)" >"$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}

die() {
  record_failure 1 "$*"
  echo "[job-exp10-scale] ERROR: $*" >&2
  exit 1
}

on_error() {
  local code=$?
  record_failure "$code" "unexpected shell failure near line ${BASH_LINENO[0]:-unknown}"
  exit "$code"
}
trap on_error ERR

case "$RESUME" in 0|1) ;; *) die "EXP10_RESUME must be 0 or 1" ;; esac
for value_name in RUN_ID ATTEMPT_ID; do
  value="${!value_name}"
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "$value_name must be a short safe identifier"
done
[[ "$NODES_PER_ARM" =~ ^[1-9][0-9]*$ ]] || die "EXP10_NODES_PER_ARM must be a positive integer"
[[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] || die "EXP10_GRAD_ACCUM must be a positive integer"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "EXP10_NUM_WORKERS must be a nonnegative integer"
[[ "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || die "EXP10_SAVE_EVERY must be a positive integer"
[[ "$WAIT_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "EXP10_WAIT_TIMEOUT_SEC must be an integer"

EFFECTIVE_BATCH=$((4 * 4 * NODES_PER_ARM * GRAD_ACCUM))
[[ "$EFFECTIVE_BATCH" == "128" ]] || die "refusing a changed statistical regime: 4 * 4 * ${NODES_PER_ARM} * ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}, expected 128"
EXPECTED_NNODES=$((NODES_PER_ARM * ${#ARMS[@]}))

[[ -d "$PROJECT_ROOT" ]] || die "repository is missing from shared deployment: $PROJECT_ROOT"
[[ -n "${TF_CONFIG:-}" ]] || die "vtraining TF_CONFIG is required for the scaled job"
cd "$PROJECT_ROOT"

# Read the company-provided global worker topology once.  We then clear
# TF_CONFIG so every later train_multinode.sh consumes the group-local torchrun
# variables exported below rather than reverting to the full 32-worker world.
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PLATFORM_NNODES="$NNODES"
PLATFORM_RANK="$NODE_RANK"
[[ "$PLATFORM_NNODES" == "$EXPECTED_NNODES" ]] || die "expected ${EXPECTED_NNODES} Workers for ${NODES_PER_ARM} nodes/arm, platform reported ${PLATFORM_NNODES}"
[[ "$PLATFORM_RANK" =~ ^[0-9]+$ && "$PLATFORM_RANK" -lt "$PLATFORM_NNODES" ]] || die "invalid platform rank: $PLATFORM_RANK"

mapfile -t PLATFORM_WORKERS < <("${JEPA_ENV}/bin/python" - <<'PY'
import json
import os
import re

workers = json.loads(os.environ["TF_CONFIG"])["cluster"]["worker"]
for worker in workers:
    print(re.sub(r":[0-9]+$", "", worker))
PY
)
[[ "${#PLATFORM_WORKERS[@]}" == "$PLATFORM_NNODES" ]] || die "TF_CONFIG worker count does not match platform topology"

GROUP_ID=$((PLATFORM_RANK / NODES_PER_ARM))
GROUP_RANK=$((PLATFORM_RANK % NODES_PER_ARM))
[[ "$GROUP_ID" -lt "${#ARMS[@]}" ]] || die "computed invalid arm group: $GROUP_ID"
ARM="${ARMS[$GROUP_ID]}"
GROUP_MASTER="${PLATFORM_WORKERS[$((GROUP_ID * NODES_PER_ARM))]}"

unset TF_CONFIG
export NPROC_PER_NODE=4 NNODES="$NODES_PER_ARM" NODE_RANK="$GROUP_RANK"
export MASTER_ADDR="$GROUP_MASTER" MASTER_PORT="${MASTER_PORT:-29500}"
export GRAD_ACCUM
export TRAIN_NPROC_PER_NODE=4 TRAIN_NNODES="$NODES_PER_ARM"
export TRAIN_NODE_RANK="$GROUP_RANK" TRAIN_MASTER_ADDR="$GROUP_MASTER"
# 128 GPUs x the historical eight PyAV workers would start 1024 readers on
# JuiceFS.  Two per GPU is a deliberate high-scale default and is overrideable
# only through the submitted job configuration.
export TRAIN_EXTRA_OVERRIDES="train.num_workers=${NUM_WORKERS}"
export EXP10_SAVE_EVERY="$SAVE_EVERY"
export RESUME

RUN_ROOT="${CLUSTER_BASE}/runs/exp10_curated/${RUN_ID}"
COORD_DIR="${RUN_ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${RUN_ROOT}/logs/${ATTEMPT_ID}"
export OUTPUT_ROOT="${RUN_ROOT}/outputs"
export RESULTS_ROOT="${RUN_ROOT}/results"
mkdir -p "$COORD_DIR" "$LOG_DIR" "$OUTPUT_ROOT" "$RESULTS_ROOT"
LOG_FILE="${LOG_DIR}/rank${PLATFORM_RANK}.log"

gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
[[ "$gpu_count" == "4" ]] || die "each Worker must expose exactly 4 GPUs, found ${gpu_count}"
echo "[job-exp10-scale] host=$(hostname) rank=${PLATFORM_RANK} group=${GROUP_ID}:${GROUP_RANK} arm=${ARM} master=${GROUP_MASTER} world=$((NODES_PER_ARM * 4)) effective_batch=${EFFECTIVE_BATCH} checkpoint_every=${SAVE_EVERY} run=${RUN_ID}"

run_stage() {
  local label="$1"
  shift
  printf '\n[job-exp10-scale] === %s ===\n' "$label" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

wait_for_marker() {
  local marker="$1" description="$2" deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
  while [[ ! -e "$marker" ]]; do
    local failures=("$COORD_DIR"/failed_rank_*)
    if [[ -e "${failures[0]}" ]]; then
      echo "[job-exp10-scale] peer failure while waiting for ${description}:" >&2
      cat "${failures[@]}" >&2 || true
      die "peer failed before ${description}"
    fi
    (( $(date +%s) < deadline )) || die "timed out after ${WAIT_TIMEOUT_SEC}s waiting for ${description}"
    sleep 30
  done
}

if [[ "$PLATFORM_RANK" == "0" ]]; then
  # Only rank 0 mutates the shared manifest. The smoke stays local 4-GPU and
  # gates correctness before 128 GPUs start distributed training.
  run_stage preflight bash scripts/direct/run_exp10_curated_4gpu.sh preflight
  run_stage prep bash scripts/direct/run_exp10_curated_4gpu.sh prep
  run_stage smoke bash scripts/direct/run_exp10_curated_4gpu.sh smoke
  touch "$COORD_DIR/gates_ready"
else
  wait_for_marker "$COORD_DIR/gates_ready" "rank-0 data gates and smoke"
fi

# All NODES_PER_ARM pods in a group execute the same arm and rendezvous through
# their group-local master.  run_exp10... forwards TRAIN_* to torchrun.
run_stage "train_${ARM}" env ONLY_ARM="$ARM" bash scripts/direct/run_exp10_curated_4gpu.sh train
if [[ "$GROUP_RANK" == "0" ]]; then
  touch "$COORD_DIR/${ARM}.done"
fi

if [[ "$PLATFORM_RANK" == "0" ]]; then
  for peer_arm in "${ARMS[@]}"; do
    wait_for_marker "$COORD_DIR/${peer_arm}.done" "checkpoint for ${peer_arm}"
  done
  run_stage eval bash scripts/direct/run_exp10_curated_4gpu.sh eval
  touch "$COORD_DIR/completed"
  echo "[job-exp10-scale] evaluation complete; entrypoint exiting so vivolm releases all Workers"
fi
