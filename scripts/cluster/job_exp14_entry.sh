#!/usr/bin/env bash
# 12 Workers -> six independent 2-Worker/8-GPU DDP worlds.
set -Eeuo pipefail

BASE="/data/vjuicefs_sz_ocr_wl/public_data/11193960"
PROJECT_ROOT="${BASE}/jepa-vlm-single-tower"
RUN_ID="${EXP14_RUN_ID:-}"
ATTEMPT_ID="${EXP14_ATTEMPT_ID:-}"
RESUME="${EXP14_RESUME:-0}"
FIXED_COMMIT="${EXP14_GIT_COMMIT:-}"
NODES_PER_ARM="${EXP14_NODES_PER_ARM:-2}"
GRAD_ACCUM="${EXP14_GRAD_ACCUM:-4}"
NUM_WORKERS="${EXP14_NUM_WORKERS:-4}"
ARMS=(b0_ce_seed1 b1_query_seed1 b2_noquery_seed0 b3_noquery_seed1 b4_query_beatcopy_seed0 b5_query_beatcopy_seed1)
EXPECTED_NNODES=$((NODES_PER_ARM * ${#ARMS[@]}))
WAIT_TIMEOUT_SEC="${EXP14_WAIT_TIMEOUT_SEC:-172800}"
COORD_DIR=""

record_failure() {
  [[ -n "$COORD_DIR" ]] || return 0
  mkdir -p "$COORD_DIR" || true
  printf 'host=%s rank=%s group=%s time=%s\n' "$(hostname)" "${PLATFORM_RANK:-?}" \
    "${GROUP_ID:-?}" "$(date -Is)" > "$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}
on_error() { code=$?; record_failure; exit "$code"; }
trap on_error ERR
die() { echo "[exp14-job] ERROR: $*" >&2; exit 1; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid run ID"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid attempt ID"
[[ "$FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXP14_GIT_COMMIT must be a full hash"
[[ -n "${TF_CONFIG:-}" ]] || die "TF_CONFIG is required"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$FIXED_COMMIT" ]] || die "checkout is not fixed commit $FIXED_COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "checkout is dirty; never pull inside a GPU Pod"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PLATFORM_NNODES="$NNODES"; PLATFORM_RANK="$NODE_RANK"
[[ "$PLATFORM_NNODES" -eq "$EXPECTED_NNODES" ]] || die "expected $EXPECTED_NNODES Workers, got $PLATFORM_NNODES"
mapfile -t WORKERS < <("${JEPA_ENV}/bin/python" - <<'PY'
import json, os, re
for value in json.loads(os.environ["TF_CONFIG"])["cluster"]["worker"]:
    print(re.sub(r":[0-9]+$", "", value))
PY
)
[[ "${#WORKERS[@]}" -eq "$PLATFORM_NNODES" ]] || die "worker topology mismatch"
GROUP_ID=$((PLATFORM_RANK / NODES_PER_ARM))
GROUP_RANK=$((PLATFORM_RANK % NODES_PER_ARM))
ARM="${ARMS[$GROUP_ID]}"
GROUP_MASTER="${WORKERS[$((GROUP_ID * NODES_PER_ARM))]}"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
[[ "$GPU_COUNT" -eq 4 ]] || die "each Worker needs 4 GPUs, found $GPU_COUNT"
WORLD_PER_ARM=$((NODES_PER_ARM * GPU_COUNT))
EFFECTIVE_BATCH=$((WORLD_PER_ARM * GRAD_ACCUM))
[[ "$EFFECTIVE_BATCH" -eq 32 ]] || die "effective batch must be 32, got $EFFECTIVE_BATCH"

unset TF_CONFIG
export NPROC_PER_NODE=4 NNODES="$NODES_PER_ARM" NODE_RANK="$GROUP_RANK"
export MASTER_ADDR="$GROUP_MASTER" MASTER_PORT="${MASTER_PORT:-29500}"
export EXP14_GRAD_ACCUM="$GRAD_ACCUM" EXP14_NUM_WORKERS="$NUM_WORKERS" EXP14_RESUME="$RESUME"
export EXP14_RUN_ROOT="${BASE}/runs/exp14/${RUN_ID}"
COORD_DIR="${EXP14_RUN_ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${EXP14_RUN_ROOT}/logs/${ATTEMPT_ID}"
mkdir -p "$COORD_DIR" "$LOG_DIR" "${EXP14_RUN_ROOT}/results/exp14_state_diagnostics"
LOG_FILE="$LOG_DIR/rank${PLATFORM_RANK}.log"

if [[ "$PLATFORM_RANK" == 0 ]]; then
  "${JEPA_ENV}/bin/python" - "$EXP14_RUN_ROOT/resource_audit.json" "$FIXED_COMMIT" <<'PY'
import json, sys
json.dump({"resource_unit":"Worker","workers":12,"gpus_per_worker":4,"total_gpus":48,
           "arms":6,"workers_per_arm":2,"gpus_per_arm":8,"gradient_accumulation":4,
           "per_device_batch":1,"effective_batch":32,"fixed_commit":sys.argv[2]},
          open(sys.argv[1], "w"), indent=2)
PY
fi
wait_for() {
  marker="$1"; deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
  while [[ ! -e "$marker" ]]; do
    failures=("$COORD_DIR"/failed_rank_*)
    [[ ! -e "${failures[0]}" ]] || { cat "${failures[@]}" >&2; die "peer failed"; }
    (( $(date +%s) < deadline )) || die "timed out waiting for $marker"
    sleep 20
  done
}
run_stage() { label="$1"; shift; echo "=== $label ===" | tee -a "$LOG_FILE"; "$@" 2>&1 | tee -a "$LOG_FILE"; }

if [[ "$PLATFORM_RANK" == 0 ]]; then
  run_stage manifest env EXP12_RUN_ROOT="$EXP14_RUN_ROOT" bash scripts/exp12/01_build_or_validate_manifest.sh
  run_stage preflight bash scripts/exp14/00_preflight.sh
  run_stage unit_tests env EXP12_RUN_ROOT="$EXP14_RUN_ROOT" bash scripts/exp12/02_run_unit_tests.sh
  if [[ "${EXP14_SKIP_SMOKE:-0}" != 1 ]]; then
    run_stage smoke bash scripts/exp14/01_smoke.sh
  fi
  touch "$COORD_DIR/gates_ready"
else
  wait_for "$COORD_DIR/gates_ready"
fi

run_stage "train_${ARM}" bash scripts/exp14/run_arm.sh "$ARM"
if [[ "$GROUP_RANK" == 0 ]]; then
  run_stage "eval400_${ARM}" bash scripts/exp14/_eval_checkpoint.sh 400 "$ARM"
  run_stage "eval800_${ARM}" bash scripts/exp14/_eval_checkpoint.sh 800 "$ARM"
  touch "$COORD_DIR/${ARM}.evaldone"
fi

if [[ "$PLATFORM_RANK" == 0 ]]; then
  for peer in "${ARMS[@]}"; do wait_for "$COORD_DIR/${peer}.evaldone"; done
  run_stage collect "${JEPA_ENV}/bin/python" scripts/exp14/02_collect_results.py \
    --root "${EXP14_RUN_ROOT}/results/exp14_state_diagnostics" \
    --source "${EXP14_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
  touch "$COORD_DIR/completed"
  echo "[exp14-job] train/eval/collect complete; exiting for automatic release"
fi
