#!/usr/bin/env bash
# Seven evaluation Workers: 3 custom-protocol partitions + 4 native protocol/task jobs.
set -Eeuo pipefail

BASE="/data/vjuicefs_sz_ocr_wl/public_data/11193960"
PROJECT_ROOT="${BASE}/jepa-vlm-single-tower"
RUN_ID="${EXP13_RUN_ID:-}"
ATTEMPT_ID="${EXP13_ATTEMPT_ID:-}"
FIXED_COMMIT="${EXP13_GIT_COMMIT:-}"
MAX_CLIPS="${EXP13_MAX_CLIPS:-0}"
EXPECTED_WORKERS=7
ROOT="${BASE}/runs/exp13/${RUN_ID}"
COORD_DIR="${ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${ROOT}/logs/${ATTEMPT_ID}"

die() { echo "[exp13-job] ERROR: $*" >&2; exit 1; }
record_failure() {
  mkdir -p "$COORD_DIR" || true
  echo "host=$(hostname) rank=${PLATFORM_RANK:-unknown} time=$(date -Is)" \
    > "$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}
on_error() { code=$?; record_failure; exit "$code"; }
trap on_error ERR

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid EXP13_RUN_ID"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid EXP13_ATTEMPT_ID"
[[ "$FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXP13_GIT_COMMIT must be a full hash"
[[ "$MAX_CLIPS" =~ ^[0-9]+$ ]] || die "EXP13_MAX_CLIPS must be nonnegative"
[[ -n "${TF_CONFIG:-}" ]] || die "TF_CONFIG is required"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$FIXED_COMMIT" ]] || die "checkout is not $FIXED_COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "checkout is dirty; never pull inside a GPU Pod"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PLATFORM_RANK="$NODE_RANK"
[[ "$NNODES" -eq "$EXPECTED_WORKERS" ]] || die "expected 7 Workers, got $NNODES"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
[[ "$GPU_COUNT" -eq 4 ]] || die "each Worker needs 4 GPUs, found $GPU_COUNT"
mkdir -p "$COORD_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/rank${PLATFORM_RANK}.log"

wait_for() {
  marker="$1"; deadline=$(( $(date +%s) + 86400 ))
  while [[ ! -e "$marker" ]]; do
    failures=("$COORD_DIR"/failed_rank_*)
    [[ ! -e "${failures[0]}" ]] || { cat "${failures[@]}" >&2; die "peer failed"; }
    (( $(date +%s) < deadline )) || die "timed out waiting for $marker"
    sleep 15
  done
}
run_stage() {
  label="$1"; shift
  echo "=== $label ===" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

export BASE PROJECT_ROOT EXP12_NATIVE_ANCHOR_ROOT="$ROOT" MAX_CLIPS GPU_LIST=0,1,2,3
if [[ "$PLATFORM_RANK" == 0 ]]; then
  run_stage preflight bash scripts/exp12/15_native_anchor_preflight.sh
  run_stage export_overlay "${JEPA_ENV}/bin/python" -m jepa_vlm.probes.native_checkpoint \
    --checkpoint "${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep/a4_ce_k64/checkpoint-800" \
    --output "$ROOT/a4_ce_k64_native_overlay.pt"
  touch "$COORD_DIR/gates_ready"
else
  wait_for "$COORD_DIR/gates_ready"
fi

if (( PLATFORM_RANK < 3 )); then
  export CUSTOM_PARTITION_INDEX="$PLATFORM_RANK" CUSTOM_PARTITION_COUNT=3
  run_stage "custom_partition_${PLATFORM_RANK}" bash scripts/exp12/16_eval_custom_anchor.sh
else
  export NATIVE_PARTITION_INDEX="$((PLATFORM_RANK - 3))" NATIVE_PARTITION_COUNT=4
  run_stage "native_partition_$((PLATFORM_RANK - 3))" bash scripts/exp12/17_eval_native_anchor.sh
fi
touch "$COORD_DIR/rank${PLATFORM_RANK}.done"

if [[ "$PLATFORM_RANK" == 0 ]]; then
  for rank in $(seq 0 $((EXPECTED_WORKERS - 1))); do
    wait_for "$COORD_DIR/rank${rank}.done"
  done
  run_stage collect bash scripts/exp12/18_collect_native_anchor.sh
  touch "$COORD_DIR/completed"
  echo "[exp13-job] complete; all commands exit and Workers are released"
fi
