#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
TEMPLATE="$PROJECT_ROOT/job_exp13_official.yaml"
DRY_RUN=0; MAX_CLIPS=0; RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --smoke) MAX_CLIPS=20 ;;
    --run-id) shift; RUN_ID="${1:?--run-id needs a value}" ;;
    *) echo "usage: $0 [--dry-run] [--smoke] [--run-id ID]" >&2; exit 2 ;;
  esac
  shift
done
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "commit all changes before submission" >&2; exit 1; }
COMMIT="$(git rev-parse HEAD)"
[[ -n "$RUN_ID" ]] || RUN_ID="official-budget-$(date +%Y%m%d-%H%M%S)-${COMMIT:0:7}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || { echo "invalid run ID" >&2; exit 2; }
ATTEMPT_ID="attempt-$(date +%Y%m%d-%H%M%S)"
TMP="$(mktemp "${TMPDIR:-/tmp}/jexp13official.XXXXXX.yaml")"; trap 'rm -f "$TMP"' EXIT
sed -e "s/EXP13_OFFICIAL_RUN_ID=unset/EXP13_OFFICIAL_RUN_ID=${RUN_ID}/" \
    -e "s/EXP13_OFFICIAL_ATTEMPT_ID=unset/EXP13_OFFICIAL_ATTEMPT_ID=${ATTEMPT_ID}/" \
    -e "s/EXP13_OFFICIAL_GIT_COMMIT=unset/EXP13_OFFICIAL_GIT_COMMIT=${COMMIT}/" \
    -e "s/EXP13_OFFICIAL_MAX_CLIPS=0/EXP13_OFFICIAL_MAX_CLIPS=${MAX_CLIPS}/" \
    "$TEMPLATE" > "$TMP"
echo "[exp13-official-submit] run=$RUN_ID attempt=$ATTEMPT_ID commit=$COMMIT max_clips=$MAX_CLIPS"
echo "[exp13-official-submit] request=4 Workers x4 L40S; evaluation only; auto-release"
if [[ "$DRY_RUN" == 1 ]]; then sed -n '1,220p' "$TMP"; exit 0; fi
[[ -x "$VTRAINING" ]] || { echo "vtraining unavailable: $VTRAINING" >&2; exit 1; }
mkdir -p "$BASE/runs/exp13-official/$RUN_ID/submission"
cp "$TMP" "$BASE/runs/exp13-official/$RUN_ID/submission/job.yaml"
"$VTRAINING" run -f "$TMP" 2>&1 | tee "$BASE/runs/exp13-official/$RUN_ID/submission/vtraining.log"
