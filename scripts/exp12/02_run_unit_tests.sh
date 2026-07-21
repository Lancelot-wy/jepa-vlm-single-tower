#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
cd "$PROJECT_ROOT"
if [[ -f scripts/cluster/env.cluster.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/env.cluster.sh
fi
PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"; PY="${PY:-python3}"
mkdir -p "${EXP12_RUN_ROOT:-outputs/exp12}/tests"
"$PY" -m pytest -q tests 2>&1 | tee "${EXP12_RUN_ROOT:-outputs/exp12}/tests/pytest.log"
