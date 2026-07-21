#!/usr/bin/env bash
# 24 Workers -> six independent 4-Worker DDP worlds; never one 96-GPU world.
set -Eeuo pipefail

BASE="/data/vjuicefs_sz_ocr_wl/public_data/11193960"
PROJECT_ROOT="${BASE}/jepa-vlm-single-tower"
RUN_ID="${EXP12_RUN_ID:-}"
ATTEMPT_ID="${EXP12_ATTEMPT_ID:-}"
RESUME="${EXP12_RESUME:-0}"
FIXED_COMMIT="${EXP12_GIT_COMMIT:-}"
NODES_PER_ARM="${EXP12_NODES_PER_ARM:-4}"
GRAD_ACCUM="${EXP12_GRAD_ACCUM:-2}"
NUM_WORKERS="${EXP12_NUM_WORKERS:-4}"
ARMS=(a0_ce_k4 a1_query_k4 a2_ce_k16 a3_query_k16 a4_ce_k64 a5_query_k64)
EXPECTED_NNODES=$((NODES_PER_ARM * ${#ARMS[@]}))
WAIT_TIMEOUT_SEC="${EXP12_WAIT_TIMEOUT_SEC:-172800}"
COORD_DIR=""

record_failure() {
  [[ -n "$COORD_DIR" ]] || return 0
  mkdir -p "$COORD_DIR" || true
  printf 'host=%s rank=%s group=%s time=%s\n' "$(hostname)" "${PLATFORM_RANK:-?}" \
    "${GROUP_ID:-?}" "$(date -Is)" > "$COORD_DIR/failed_rank_${PLATFORM_RANK:-unknown}" || true
}
on_error() { code=$?; record_failure; exit "$code"; }
trap on_error ERR
die() { echo "[exp12-job] ERROR: $*" >&2; exit 1; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid run ID"
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || die "invalid attempt ID"
[[ "$FIXED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXP12_GIT_COMMIT must be a full hash"
[[ -n "${TF_CONFIG:-}" ]] || die "TF_CONFIG is required"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$FIXED_COMMIT" ]] || die "GPU Pod checkout is not fixed commit $FIXED_COMMIT"
[[ -z "$(git status --porcelain)" ]] || die "GPU Pod checkout is dirty; never pull inside the Pod"

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
WORLD_PER_ARM=$((NODES_PER_ARM * 4))
EFFECTIVE_BATCH=$((WORLD_PER_ARM * 1 * GRAD_ACCUM))
[[ "$EFFECTIVE_BATCH" -eq 32 ]] || die "effective batch must be 32, got $EFFECTIVE_BATCH"

unset TF_CONFIG
export NPROC_PER_NODE=4 NNODES="$NODES_PER_ARM" NODE_RANK="$GROUP_RANK"
export MASTER_ADDR="$GROUP_MASTER" MASTER_PORT="${MASTER_PORT:-29500}"
export EXP12_GRAD_ACCUM="$GRAD_ACCUM" EXP12_NUM_WORKERS="$NUM_WORKERS" EXP12_RESUME="$RESUME"
export EXP12_RUN_ROOT="${BASE}/runs/exp12/${RUN_ID}"
COORD_DIR="${EXP12_RUN_ROOT}/coord/${ATTEMPT_ID}"
LOG_DIR="${EXP12_RUN_ROOT}/logs/${ATTEMPT_ID}"
mkdir -p "$COORD_DIR" "$LOG_DIR" "${EXP12_RUN_ROOT}/results/exp12_orca_token_sweep"
LOG_FILE="$LOG_DIR/rank${PLATFORM_RANK}.log"

if [[ "$PLATFORM_RANK" == 0 ]]; then
  cat > "${EXP12_RUN_ROOT}/resource_audit.json" <<JSON
{"resource_unit":"Worker","workers":24,"gpus_per_worker":4,"total_gpus":96,
 "arms":6,"workers_per_arm":4,"gpus_per_arm":16,"gradient_accumulation":2,
 "per_device_batch":1,"effective_batch":32,"fixed_commit":"${FIXED_COMMIT}"}
JSON
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
  run_stage manifest bash scripts/exp12/01_build_or_validate_manifest.sh
  run_stage preflight bash scripts/exp12/00_preflight.sh
  run_stage unit_tests bash scripts/exp12/02_run_unit_tests.sh
  run_stage smoke_all bash scripts/exp12/06_smoke_all.sh
  touch "$COORD_DIR/gates_ready"
else
  wait_for "$COORD_DIR/gates_ready"
fi

run_stage "train_${ARM}" bash scripts/exp12/run_arm.sh "$ARM"
[[ "$GROUP_RANK" == 0 ]] && touch "$COORD_DIR/${ARM}.done"

if [[ "$GROUP_RANK" == 0 ]]; then
  # Six arm leaders evaluate concurrently on one GPU each.  Target/worker
  # followers can exit after torchrun instead of leaving 95 GPUs idle behind a
  # single-node sequential evaluator.
  run_stage "eval400_${ARM}" bash scripts/exp12/_eval_checkpoint.sh 400 "$ARM"
  run_stage "eval800_${ARM}" bash scripts/exp12/_eval_checkpoint.sh 800 "$ARM"
  touch "$COORD_DIR/${ARM}.evaldone"
fi

if [[ "$PLATFORM_RANK" == 0 ]]; then
  for peer in "${ARMS[@]}"; do wait_for "$COORD_DIR/${peer}.evaldone"; done
  run_stage collect "${JEPA_ENV}/bin/python" scripts/exp12/12_collect_results.py \
    --root "${EXP12_RUN_ROOT}/results/exp12_orca_token_sweep"
  run_stage select "${JEPA_ENV}/bin/python" scripts/exp12/13_select_best_k.py \
    --comparison "${EXP12_RUN_ROOT}/results/exp12_orca_token_sweep/comparison.json"
  touch "$COORD_DIR/completed"
  echo "[exp12-job] training, evaluation, collection complete; exiting for automatic release"
fi
