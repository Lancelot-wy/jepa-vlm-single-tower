#!/usr/bin/env bash
# Four-pod vivolm entrypoint for the current EXP-10 experiment.
#
# The job YAML requests four 4xL40S Workers.  The nodes coordinate only through
# a run-specific directory on shared JuiceFS: rank 0 first owns the common data
# gates and two-arm smoke; then every rank launches exactly one independent
# single-node / 4-GPU arm.  Rank 0 evaluates after all four checkpoints exist.

set -Eeuo pipefail

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
RUN_ID="${EXP10_RUN_ID:-}"
ATTEMPT_ID="${EXP10_ATTEMPT_ID:-}"
RESUME="${EXP10_RESUME:-0}"
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
  printf 'code=%s host=%s rank=%s message=%s time=%s\n' \
    "$code" "$(hostname)" "${PLATFORM_RANK:-unknown}" "$message" "$(date -Is)" \
    >"$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}

die() {
  record_failure 1 "$*"
  echo "[job-exp10] ERROR: $*" >&2
  exit 1
}

on_error() {
  local code=$?
  record_failure "$code" "unexpected shell failure near line ${BASH_LINENO[0]:-unknown}"
  exit "$code"
}
trap on_error ERR

case "$RESUME" in
  0|1) ;;
  *) die "EXP10_RESUME must be 0 or 1, got: $RESUME" ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "EXP10_RUN_ID must be a short safe identifier, got: ${RUN_ID:-<empty>}"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "EXP10_ATTEMPT_ID must be a short safe identifier, got: ${ATTEMPT_ID:-<empty>}"
[[ "$WAIT_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "EXP10_WAIT_TIMEOUT_SEC must be an integer"

[[ -d "$PROJECT_ROOT" ]] || die "repository is missing from the shared deployment path: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

# Read the platform rendezvous before clearing TF_CONFIG.  The actual arms must
# be four independent *single-node* torchrun jobs, not one accidental 16-GPU
# DDP job; direct runner calls below therefore run with TF_CONFIG unset.
PLATFORM_NNODES="$NNODES"
PLATFORM_RANK="$NODE_RANK"
[[ "$PLATFORM_NNODES" == "4" ]] || die "EXP-10 parallel job requires Worker.num=4, platform reported NNODES=${PLATFORM_NNODES}"
[[ "$PLATFORM_RANK" =~ ^[0-3]$ ]] || die "expected platform rank 0..3, got ${PLATFORM_RANK}"
ARM="${ARMS[$PLATFORM_RANK]}"

unset TF_CONFIG
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE=4

RUN_ROOT="${CLUSTER_BASE}/runs/exp10_curated/${RUN_ID}"
COORD_DIR="${RUN_ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${RUN_ROOT}/logs/${ATTEMPT_ID}"
export OUTPUT_ROOT="${RUN_ROOT}/outputs"
export RESULTS_ROOT="${RUN_ROOT}/results"
mkdir -p "$COORD_DIR" "$LOG_DIR" "$OUTPUT_ROOT" "$RESULTS_ROOT"
LOG_FILE="${LOG_DIR}/rank${PLATFORM_RANK}.log"

echo "[job-exp10] host=$(hostname) rank=${PLATFORM_RANK} arm=${ARM} run_id=${RUN_ID} attempt=${ATTEMPT_ID} resume=${RESUME} log=${LOG_FILE}"
echo "[job-exp10] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[job-exp10] visible_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"

export RESUME

run_stage() {
  local label="$1"
  shift
  printf '\n[job-exp10] === %s ===\n' "$label" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

wait_for_marker() {
  local marker="$1" description="$2" deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
  while [[ ! -e "$marker" ]]; do
    local failures=("$COORD_DIR"/failed_rank_*)
    if [[ -e "${failures[0]}" ]]; then
      echo "[job-exp10] peer failure while waiting for ${description}:" >&2
      cat "${failures[@]}" >&2 || true
      die "peer failed before ${description}"
    fi
    (( $(date +%s) < deadline )) || die "timed out after ${WAIT_TIMEOUT_SEC}s waiting for ${description}"
    sleep 30
  done
}

if [[ "$PLATFORM_RANK" == "0" ]]; then
  # These are the only shared mutation phases.  Other nodes wait for a durable
  # ready marker rather than racing to build/remove the same manifest.
  run_stage preflight bash scripts/direct/run_exp10_curated_4gpu.sh preflight
  run_stage prep bash scripts/direct/run_exp10_curated_4gpu.sh prep
  run_stage smoke bash scripts/direct/run_exp10_curated_4gpu.sh smoke
  touch "$COORD_DIR/gates_ready"
else
  wait_for_marker "$COORD_DIR/gates_ready" "rank-0 data gates and smoke"
fi

run_stage "train_${ARM}" env ONLY_ARM="$ARM" bash scripts/direct/run_exp10_curated_4gpu.sh train
touch "$COORD_DIR/${ARM}.done"

if [[ "$PLATFORM_RANK" == "0" ]]; then
  for peer_arm in "${ARMS[@]}"; do
    wait_for_marker "$COORD_DIR/${peer_arm}.done" "checkpoint for ${peer_arm}"
  done
  run_stage eval bash scripts/direct/run_exp10_curated_4gpu.sh eval
  touch "$COORD_DIR/completed"
  echo "[job-exp10] completed successfully: ${RUN_ROOT}"
fi
