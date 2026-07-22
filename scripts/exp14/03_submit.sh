#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
TEMPLATE="$PROJECT_ROOT/job_exp14.yaml"
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
[[ -z "$(git status --porcelain)" ]] || { echo "commit all EXP-14 changes first" >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"
[[ "$RESUME" == 0 || "$RUN_ID_PROVIDED" == 1 ]] || { echo "resume needs --run-id" >&2; exit 1; }
[[ -n "$RUN_ID" ]] || RUN_ID="exp14-$(date +%Y%m%d-%H%M%S)-${COMMIT:0:7}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || { echo "invalid run ID" >&2; exit 2; }
ATTEMPT_ID="attempt-$(date +%Y%m%d-%H%M%S)"
TMP="$(mktemp "${TMPDIR:-/tmp}/jexp14.XXXXXX.yaml")"; trap 'rm -f "$TMP"' EXIT
sed -e "s/EXP14_RUN_ID=unset/EXP14_RUN_ID=${RUN_ID}/" \
    -e "s/EXP14_ATTEMPT_ID=unset/EXP14_ATTEMPT_ID=${ATTEMPT_ID}/" \
    -e "s/EXP14_RESUME=0/EXP14_RESUME=${RESUME}/" \
    -e "s/EXP14_GIT_COMMIT=unset/EXP14_GIT_COMMIT=${COMMIT}/" "$TEMPLATE" > "$TMP"
echo "[exp14-submit] run=$RUN_ID attempt=$ATTEMPT_ID commit=$COMMIT"
echo "[exp14-submit] request=12 Workers x4 L40S; 6 worlds x8 GPUs; GA=4; EB=32"
if [[ "$DRY_RUN" == 1 ]]; then sed -n '1,220p' "$TMP"; exit 0; fi
[[ -x "$VTRAINING" ]] || { echo "vtraining unavailable: $VTRAINING" >&2; exit 1; }
mkdir -p "$BASE/runs/exp14/$RUN_ID/submission"
cp "$TMP" "$BASE/runs/exp14/$RUN_ID/submission/job.yaml"
"$VTRAINING" run -f "$TMP" 2>&1 | tee "$BASE/runs/exp14/$RUN_ID/submission/vtraining.log"
