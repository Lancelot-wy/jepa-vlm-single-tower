#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
TEMPLATE="${PROJECT_ROOT}/job_exp12.yaml"
RESUME=0; DRY_RUN=0; RUN_ID=""; RUN_ID_PROVIDED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --run-id) shift; RUN_ID="${1:?--run-id needs a value}"; RUN_ID_PROVIDED=1 ;;
    *) echo "usage: $0 [--dry-run] [--resume --run-id ID]" >&2; exit 2 ;;
  esac
  shift
done
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "commit all EXP-12 changes first" >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"
[[ "$RESUME" == 0 || "$RUN_ID_PROVIDED" == 1 ]] || {
  echo "resume requires an existing --run-id" >&2; exit 1;
}
[[ -n "$RUN_ID" ]] || RUN_ID="exp12-$(date +%Y%m%d-%H%M%S)-${COMMIT:0:7}"
ATTEMPT_ID="attempt-$(date +%Y%m%d-%H%M%S)"
TMP="$(mktemp "${TMPDIR:-/tmp}/jexp12.XXXXXX.yaml")"; trap 'rm -f "$TMP"' EXIT
sed -e "s/EXP12_RUN_ID=unset/EXP12_RUN_ID=${RUN_ID}/" \
    -e "s/EXP12_ATTEMPT_ID=unset/EXP12_ATTEMPT_ID=${ATTEMPT_ID}/" \
    -e "s/EXP12_RESUME=0/EXP12_RESUME=${RESUME}/" \
    -e "s/EXP12_GIT_COMMIT=unset/EXP12_GIT_COMMIT=${COMMIT}/" "$TEMPLATE" > "$TMP"
echo "[exp12-submit] run=$RUN_ID commit=$COMMIT"
echo "[exp12-submit] actual company request: 24 Workers x4 GPUs; six worlds x16 GPUs; GA=2; EB=32"
for arm in a0_ce_k4 a1_query_k4 a2_ce_k16 a3_query_k16 a4_ce_k64 a5_query_k64; do
  echo "logical_job_id=${RUN_ID}-${arm}"
done
if [[ "$DRY_RUN" == 1 ]]; then cat "$TMP"; exit 0; fi
[[ -x "$VTRAINING" ]] || { echo "vtraining unavailable: $VTRAINING" >&2; exit 1; }
mkdir -p "${BASE}/runs/exp12/${RUN_ID}/submission"
cp "$TMP" "${BASE}/runs/exp12/${RUN_ID}/submission/job.yaml"
"$VTRAINING" run -f "$TMP" 2>&1 | tee "${BASE}/runs/exp12/${RUN_ID}/submission/vtraining.log"
