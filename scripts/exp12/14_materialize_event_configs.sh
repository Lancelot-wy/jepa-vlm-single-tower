#!/usr/bin/env bash
set -Eeuo pipefail
BEST_K="${1:?usage: $0 4|16|64}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
cd "$PROJECT_ROOT"
PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"; PY="${PY:-python3}"
exec "$PY" scripts/exp12/materialize_event_configs.py --best-k "$BEST_K"
