#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
RUN_ID="${1:?usage: $0 <run-id>}"
ROOT="$BASE/runs/exp14/$RUN_ID"
echo "root=$ROOT"
find "$ROOT/coord" -maxdepth 2 -type f -print 2>/dev/null | sort || true
find "$ROOT/results/exp14_state_diagnostics" -maxdepth 2 \
  \( -name 'checkpoint_meta.json' -o -name 'scorecard.json' -o -name 'comparison.json' \) \
  -print 2>/dev/null | sort || true
for log in "$ROOT"/logs/*/rank0.log; do [[ -f "$log" ]] && tail -n 80 "$log"; done
