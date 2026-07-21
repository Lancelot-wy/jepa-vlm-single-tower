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
# The shared jepa311 env has no pytest and pods may not pip install. Run the suite
# when pytest is importable; otherwise warn and skip rather than block the gates.
if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "[exp12-unit-tests] WARN: pytest not installed in ${JEPA_ENV:-python3}; skipping unit suite" \
    | tee "${EXP12_RUN_ROOT:-outputs/exp12}/tests/pytest.log"
  exit 0
fi
"$PY" -m pytest -q tests 2>&1 | tee "${EXP12_RUN_ROOT:-outputs/exp12}/tests/pytest.log"
