#!/usr/bin/env bash
# Read-only status summary. Usage: inspect_exp11_run.sh <run-id>
set -euo pipefail
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
RUN_ID="${1:?usage: $0 <run-id>}"
ROOT="${BASE}/runs/exp11_orca/${RUN_ID}"
CONTROL_DIR="${EXP11_CONTROL_DIR:-${BASE}/outputs/exp11_frozen_sft_s0}"
ARMS=(exp11_mask15_s0 exp11_orca_noquery_s0 exp11_orca_obs_s0)
[[ -d "$ROOT" ]] || { echo "missing run: $ROOT" >&2; exit 1; }
echo "run_id=${RUN_ID} root=${ROOT}"
echo "[exp11_frozen_sft_s0] reused_control=${CONTROL_DIR} checkpoint=$([[ -f "$CONTROL_DIR/step_1000/state.pt" ]] && echo present || echo missing)"
for arm in "${ARMS[@]}"; do
  log="$ROOT/outputs/$arm/log.jsonl"
  ckpts="$(find "$ROOT/outputs/$arm" -maxdepth 2 -name state.pt 2>/dev/null | sort -V | tail -3 | tr '\n' ' ')"
  echo "[$arm] checkpoints=${ckpts:-none}"
  [[ -f "$log" ]] && tail -1 "$log" || true
done
[[ -f "$ROOT/results/scorecard.json" ]] && cat "$ROOT/results/scorecard.json" || echo "scorecard=pending"
find "$ROOT/coord" -maxdepth 2 -type f -print 2>/dev/null | sort || true
