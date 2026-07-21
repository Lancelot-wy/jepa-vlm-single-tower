#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
RUN_ID="${1:?usage: $0 <run-id>}"
ROOT="${BASE}/runs/exp12/${RUN_ID}"
[[ -d "$ROOT" ]] || { echo "missing run $ROOT" >&2; exit 1; }
echo "run_id=$RUN_ID root=$ROOT"
[[ -f "$ROOT/resource_audit.json" ]] && cat "$ROOT/resource_audit.json"
for arm in a0_ce_k4 a1_query_k4 a2_ce_k16 a3_query_k16 a4_ce_k64 a5_query_k64; do
  out="$ROOT/results/exp12_orca_token_sweep/$arm"
  checkpoint="missing"; [[ -f "$out/checkpoint-800/state.pt" ]] && checkpoint="complete"
  echo "[$arm] checkpoint800=$checkpoint"
  [[ -f "$out/trainer_log.jsonl" ]] && tail -1 "$out/trainer_log.jsonl" || true
done
find "$ROOT/coord" -maxdepth 2 -type f -print 2>/dev/null | sort || true
[[ -f "$ROOT/results/exp12_orca_token_sweep/comparison.md" ]] && \
  cat "$ROOT/results/exp12_orca_token_sweep/comparison.md"
