#!/usr/bin/env bash
# EXP-11 entrypoint: N independent DDP groups, one per arm named in EXP11_ARMS.
# Each group trains a frozen-ViT arm to EXP11_MAX_STEPS updates; rank 0 runs the
# shared data/smoke gates and, once every arm in this job has a final checkpoint,
# evaluates them all on MVBench + TempCompass.

set -Eeuo pipefail

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
RUN_ID="${EXP11_RUN_ID:-}"
ATTEMPT_ID="${EXP11_ATTEMPT_ID:-}"
RESUME="${EXP11_RESUME:-0}"
NODES_PER_ARM="${EXP11_NODES_PER_ARM:-8}"
GRAD_ACCUM="${EXP11_GRAD_ACCUM:-1}"
NUM_WORKERS="${EXP11_NUM_WORKERS:-2}"
MAX_STEPS="${EXP11_MAX_STEPS:-4000}"
SAVE_EVERY="${EXP11_SAVE_EVERY:-250}"
WAIT_TIMEOUT_SEC="${EXP11_WAIT_TIMEOUT_SEC:-172800}"
ARMS_CSV="${EXP11_ARMS:-exp11_frozen_sft_s0,exp11_mask15_s0,exp11_orca_noquery_s0,exp11_orca_obs_s0}"
IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
COORD_DIR=""

record_failure() {
  local code="${1:-1}" message="${2:-unspecified failure}"
  [[ -n "$COORD_DIR" ]] || return 0
  mkdir -p "$COORD_DIR" || true
  printf 'code=%s host=%s rank=%s group=%s message=%s time=%s\n' \
    "$code" "$(hostname)" "${PLATFORM_RANK:-unknown}" "${GROUP_ID:-unknown}" \
    "$message" "$(date -Is)" >"$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}
die() { record_failure 1 "$*"; echo "[job-exp11] ERROR: $*" >&2; exit 1; }
on_error() { local code=$?; record_failure "$code" "shell failure near line ${BASH_LINENO[0]:-unknown}"; exit "$code"; }
trap on_error ERR

case "$RESUME" in 0|1) ;; *) die "EXP11_RESUME must be 0 or 1" ;; esac
for value_name in RUN_ID ATTEMPT_ID; do
  value="${!value_name}"
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "$value_name must be a short safe identifier"
done
for value_name in NODES_PER_ARM GRAD_ACCUM MAX_STEPS SAVE_EVERY; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$value_name must be positive"
done
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be nonnegative"
[[ "${#ARMS[@]}" -ge 1 ]] || die "EXP11_ARMS is empty"

EFFECTIVE_BATCH=$((4 * 4 * NODES_PER_ARM * GRAD_ACCUM))
# Hold 128 when topology allows; Job A uses 15 nodes (3 arms x 5 nodes x GA2)
# because mm-general only had 15 schedulable nodes, so 160 is permitted there.
# The 24-node layout runs 4 arms x 6 nodes x GA1 = 96, so 96 is permitted too.
case "$EFFECTIVE_BATCH" in
  96|128|160) ;;
  *) die "expected effective batch 96, 128 or 160, got ${EFFECTIVE_BATCH}" ;;
esac
EXPECTED_NNODES=$((NODES_PER_ARM * ${#ARMS[@]}))
[[ -d "$PROJECT_ROOT" ]] || die "repository missing: $PROJECT_ROOT"
[[ -n "${TF_CONFIG:-}" ]] || die "vtraining TF_CONFIG is required"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PLATFORM_NNODES="$NNODES"
PLATFORM_RANK="$NODE_RANK"
[[ "$PLATFORM_NNODES" == "$EXPECTED_NNODES" ]] || die "expected ${EXPECTED_NNODES} Workers, platform reported ${PLATFORM_NNODES}"

mapfile -t PLATFORM_WORKERS < <("${JEPA_ENV}/bin/python" - <<'PY'
import json, os, re
for worker in json.loads(os.environ["TF_CONFIG"])["cluster"]["worker"]:
    print(re.sub(r":[0-9]+$", "", worker))
PY
)
[[ "${#PLATFORM_WORKERS[@]}" == "$PLATFORM_NNODES" ]] || die "TF_CONFIG topology mismatch"

GROUP_ID=$((PLATFORM_RANK / NODES_PER_ARM))
GROUP_RANK=$((PLATFORM_RANK % NODES_PER_ARM))
[[ "$GROUP_ID" -lt "${#ARMS[@]}" ]] || die "invalid group ${GROUP_ID}"
ARM="${ARMS[$GROUP_ID]}"
GROUP_MASTER="${PLATFORM_WORKERS[$((GROUP_ID * NODES_PER_ARM))]}"

unset TF_CONFIG
export NPROC_PER_NODE=4 NNODES="$NODES_PER_ARM" NODE_RANK="$GROUP_RANK"
export MASTER_ADDR="$GROUP_MASTER" MASTER_PORT="${MASTER_PORT:-29500}"
export GRAD_ACCUM
export TRAIN_NPROC_PER_NODE=4 TRAIN_NNODES="$NODES_PER_ARM"
export TRAIN_NODE_RANK="$GROUP_RANK" TRAIN_MASTER_ADDR="$GROUP_MASTER"
export TRAIN_EXTRA_OVERRIDES="train.num_workers=${NUM_WORKERS}"
export EXP11_ARMS="$ARMS_CSV" EXP11_NODES_PER_ARM="$NODES_PER_ARM"
export EXP11_MAX_STEPS="$MAX_STEPS" EXP11_SAVE_EVERY="$SAVE_EVERY" RESUME

RUN_ROOT="${CLUSTER_BASE}/runs/exp11_orca/${RUN_ID}"
COORD_DIR="${RUN_ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${RUN_ROOT}/logs/${ATTEMPT_ID}"
export OUTPUT_ROOT="${RUN_ROOT}/outputs" RESULTS_ROOT="${RUN_ROOT}/results"
mkdir -p "$COORD_DIR" "$LOG_DIR" "$OUTPUT_ROOT" "$RESULTS_ROOT"
LOG_FILE="${LOG_DIR}/rank${PLATFORM_RANK}.log"

gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
[[ "$gpu_count" == 4 ]] || die "each Worker must expose 4 GPUs, found ${gpu_count}"
echo "[job-exp11] host=$(hostname) rank=${PLATFORM_RANK} group=${GROUP_ID}:${GROUP_RANK} arm=${ARM} arms=[${ARMS_CSV}] world=$((NODES_PER_ARM * 4)) batch=${EFFECTIVE_BATCH} steps=${MAX_STEPS} run=${RUN_ID}"

run_stage() {
  local label="$1"; shift
  printf '\n[job-exp11] === %s ===\n' "$label" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}
wait_for_marker() {
  local marker="$1" description="$2" deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
  while [[ ! -e "$marker" ]]; do
    local failures=("$COORD_DIR"/failed_rank_*)
    if [[ -e "${failures[0]}" ]]; then
      cat "${failures[@]}" >&2 || true
      die "peer failed before ${description}"
    fi
    (( $(date +%s) < deadline )) || die "timed out waiting for ${description}"
    sleep 30
  done
}

if [[ "$PLATFORM_RANK" == 0 ]]; then
  # Reuse the already audited/fingerprinted EXP-10 four-source manifest builder.
  run_stage data_prep bash scripts/direct/run_exp10_curated_4gpu.sh prep
  run_stage exp11_preflight bash scripts/direct/run_exp11_orca_pilot.sh preflight
  run_stage smoke bash scripts/direct/run_exp11_orca_pilot.sh smoke
  touch "$COORD_DIR/gates_ready"
else
  wait_for_marker "$COORD_DIR/gates_ready" "data gates and arm smoke"
fi

run_stage "train_${ARM}" env ONLY_ARM="$ARM" bash scripts/direct/run_exp11_orca_pilot.sh train
if [[ "$GROUP_RANK" == 0 ]]; then touch "$COORD_DIR/${ARM}.done"; fi

if [[ "$PLATFORM_RANK" == 0 ]]; then
  for peer_arm in "${ARMS[@]}"; do wait_for_marker "$COORD_DIR/${peer_arm}.done" "${peer_arm} checkpoint"; done
  run_stage eval bash scripts/direct/run_exp11_orca_pilot.sh eval
  touch "$COORD_DIR/completed"
  echo "[job-exp11] training and evaluation complete; exiting to release all Workers"
fi
