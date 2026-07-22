#!/usr/bin/env bash
# Four Workers, one budget/model protocol per Worker, four GPU shards each.
set -Eeuo pipefail

BASE="/data/vjuicefs_sz_ocr_wl/public_data/11193960"
PROJECT_ROOT="${BASE}/jepa-vlm-single-tower"
RUN_ID="${EXP13_OFFICIAL_RUN_ID:-}"
ATTEMPT_ID="${EXP13_OFFICIAL_ATTEMPT_ID:-}"
FIXED_COMMIT="${EXP13_OFFICIAL_GIT_COMMIT:-}"
MAX_CLIPS="${EXP13_OFFICIAL_MAX_CLIPS:-0}"
EXPECTED_WORKERS=4
ROOT="${BASE}/runs/exp13-official/${RUN_ID}"
COORD_DIR="$ROOT/coord/$ATTEMPT_ID"
LOG_DIR="$ROOT/logs/$ATTEMPT_ID"

die() { echo "[exp13-official-job] ERROR: $*" >&2; exit 1; }
record_failure() {
  mkdir -p "$COORD_DIR" || true
  echo "host=$(hostname) rank=${PLATFORM_RANK:-unknown} time=$(date -Is)" \
    > "$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}
on_error() { code=$?; record_failure; exit "$code"; }
trap on_error ERR
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid run ID"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid attempt ID"
[[ "$FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "commit must be a full hash"
[[ "$MAX_CLIPS" =~ ^[0-9]+$ ]] || die "max clips must be nonnegative"
[[ -n "${TF_CONFIG:-}" ]] || die "TF_CONFIG is required"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$FIXED_COMMIT" ]] || die "checkout is not $FIXED_COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "checkout is dirty"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PLATFORM_RANK="$NODE_RANK"
[[ "$NNODES" -eq "$EXPECTED_WORKERS" ]] || die "expected 4 Workers, got $NNODES"
[[ "$(nvidia-smi -L | grep -c '^GPU ' || true)" -eq 4 ]] || die "each Worker needs 4 GPUs"
mkdir -p "$COORD_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/rank${PLATFORM_RANK}.log"

wait_for() {
  marker="$1"; deadline=$(( $(date +%s) + 172800 ))
  while [[ ! -e "$marker" ]]; do
    failures=("$COORD_DIR"/failed_rank_*)
    [[ ! -e "${failures[0]}" ]] || { cat "${failures[@]}" >&2; die "peer failed"; }
    (( $(date +%s) < deadline )) || die "timed out waiting for $marker"
    sleep 15
  done
}
run_stage() { label="$1"; shift; echo "=== $label ===" | tee -a "$LOG_FILE"; "$@" 2>&1 | tee -a "$LOG_FILE"; }

export BASE PROJECT_ROOT EXP13_OFFICIAL_ROOT="$ROOT" EXP13_OFFICIAL_MAX_CLIPS="$MAX_CLIPS"
export EXP13_OFFICIAL_GIT_COMMIT="$FIXED_COMMIT" GPU_LIST=0,1,2,3
if [[ "$PLATFORM_RANK" == 0 ]]; then
  run_stage preflight bash scripts/exp13/00_official_preflight.sh
  run_stage export_overlay "${JEPA_ENV}/bin/python" -m jepa_vlm.probes.native_checkpoint \
    --checkpoint "${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep/a4_ce_k64/checkpoint-800" \
    --output "$ROOT/a4_ce_k64_native_overlay.pt"
  "${JEPA_ENV}/bin/python" - "$ROOT/resource_audit.json" "$FIXED_COMMIT" <<'PY'
import json, sys
json.dump({"resource_unit":"Worker","workers":4,"gpus_per_worker":4,"total_gpus":16,
           "protocols":4,"gpu_shards_per_protocol":4,"fixed_commit":sys.argv[2],
           "release":"automatic_on_entrypoint_exit"}, open(sys.argv[1], "w"), indent=2)
PY
  touch "$COORD_DIR/gates_ready"
else
  wait_for "$COORD_DIR/gates_ready"
fi

export OFFICIAL_PARTITION_INDEX="$PLATFORM_RANK" OFFICIAL_PARTITION_COUNT=4
run_stage "official_partition_${PLATFORM_RANK}" bash scripts/exp13/01_eval_official.sh
touch "$COORD_DIR/rank${PLATFORM_RANK}.done"
if [[ "$PLATFORM_RANK" == 0 ]]; then
  for rank in 0 1 2 3; do wait_for "$COORD_DIR/rank${rank}.done"; done
  run_stage collect "${JEPA_ENV}/bin/python" scripts/exp13/02_collect_official.py --root "$ROOT"
  touch "$COORD_DIR/completed"
  echo "[exp13-official-job] complete; Workers are released on exit"
fi
