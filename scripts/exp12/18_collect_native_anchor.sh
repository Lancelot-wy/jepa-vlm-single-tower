#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
ROOT="${EXP12_NATIVE_ANCHOR_ROOT:-${BASE}/runs/exp13/native-anchor}"
cd "$PROJECT_ROOT"
if [[ -f scripts/cluster/env.cluster.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/env.cluster.sh
fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"
PY="${PY:-$(command -v python3)}"
exec "$PY" scripts/exp12/18_collect_native_anchor.py --root "$ROOT" "$@"
