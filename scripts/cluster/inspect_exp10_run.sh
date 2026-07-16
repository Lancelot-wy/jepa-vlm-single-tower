#!/usr/bin/env bash
# Read-only shared-filesystem status summary for a submitted EXP-10 run.
# Usage: bash scripts/cluster/inspect_exp10_run.sh <run-id>

set -euo pipefail

RUN_ID="${1:-}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
ROOT="${BASE}/runs/exp10_curated/${RUN_ID}"
ARMS=(
  exp10_curated_sft_s0
  exp10_curated_mse_s0
  exp10_curated_sft_s1
  exp10_curated_mse_s1
)

die() { echo "[inspect-exp10] ERROR: $*" >&2; exit 1; }
[[ -n "$RUN_ID" ]] || die "usage: $0 <run-id>"
[[ -d "$ROOT" ]] || die "run root does not exist: $ROOT"

latest_step() {
  local arm_root="$1" state step best=-1
  for state in "$arm_root"/step_*/state.pt; do
    [[ -f "$state" ]] || continue
    step="${state%/state.pt}"
    step="${step##*/step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step > best )) && best="$step"
  done
  printf '%s' "$best"
}

echo "[inspect-exp10] run_id=${RUN_ID} root=${ROOT}"
for arm in "${ARMS[@]}"; do
  arm_root="${ROOT}/outputs/${arm}"
  step="$(latest_step "$arm_root")"
  if [[ -f "${arm_root}/step_4000/state.pt" ]]; then
    printf '  %-26s complete (step_4000)\n' "$arm"
  elif (( step >= 0 )); then
    printf '  %-26s partial (latest step_%s)\n' "$arm" "$step"
  else
    printf '  %-26s not started\n' "$arm"
  fi
done

echo "[inspect-exp10] coordination attempts:"
found_attempt=0
for attempt in "$ROOT"/coord/*; do
  [[ -d "$attempt" ]] || continue
  found_attempt=1
  name="${attempt##*/}"
  printf '  %s:' "$name"
  [[ -f "$attempt/gates_ready" ]] && printf ' gates_ready'
  [[ -f "$attempt/completed" ]] && printf ' completed'
  failures=("$attempt"/failed_rank_*)
  [[ -f "${failures[0]}" ]] && printf ' FAILED'
  printf '\n'
  if [[ -f "${failures[0]}" ]]; then
    cat "${failures[@]}"
  fi
done
(( found_attempt )) || echo "  (no worker has initialized its shared coordination directory yet)"

if [[ -f "$ROOT/results/scorecard.json" ]]; then
  echo "[inspect-exp10] evaluation scorecard:"
  cat "$ROOT/results/scorecard.json"
else
  echo "[inspect-exp10] evaluation scorecard not present yet"
fi

echo "[inspect-exp10] logs: ${ROOT}/logs/<attempt-id>/rank{0,1,2,3}.log"
