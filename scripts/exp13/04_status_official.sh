#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
RUN_ID="${1:?usage: $0 <run-id>}"
ROOT="$BASE/runs/exp13-official/$RUN_ID"
echo "root=$ROOT"
find "$ROOT/coord" -maxdepth 2 -type f -print 2>/dev/null | sort || true
find "$ROOT" -maxdepth 1 -type f -print 2>/dev/null | sort || true
for log in "$ROOT"/logs/*/rank0.log; do [[ -f "$log" ]] && tail -n 80 "$log"; done
