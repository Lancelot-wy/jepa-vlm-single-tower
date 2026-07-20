#!/usr/bin/env bash
# Submit/resume the 24-Worker EXP-11 overnight pilot.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
JOB_TEMPLATE="${PROJECT_ROOT}/job_exp11_orca24.yaml"
RESUME=0
DRY_RUN=0
RUN_ID=""
RUN_ID_EXPLICIT=0

die() { echo "[submit-exp11] ERROR: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --run-id) shift; [[ $# -gt 0 ]] || die "--run-id requires a value"; RUN_ID="$1"; RUN_ID_EXPLICIT=1 ;;
    -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
    *) die "usage: $0 [--dry-run] [--resume --run-id <existing-run-id>]" ;;
  esac
  shift
done

[[ -d "$PROJECT_ROOT/.git" ]] || die "expected Git checkout at $PROJECT_ROOT"
[[ -f "$JOB_TEMPLATE" ]] || die "missing $JOB_TEMPLATE"
[[ -x "$PROJECT_ROOT/scripts/cluster/job_exp11_orca24_entry.sh" ]] || die "entrypoint is not executable"
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || die "checkout has local changes; pull a committed revision"
HEAD="$(git rev-parse --short HEAD)"
[[ -n "$RUN_ID" ]] || RUN_ID="exp11-orca-$(date '+%Y%m%d-%H%M%S')-${HEAD}"
ATTEMPT_ID="attempt-$(date '+%Y%m%d-%H%M%S')"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "invalid run id"
if [[ "$RESUME" == 1 && "$RUN_ID_EXPLICIT" != 1 ]]; then die "--resume requires --run-id"; fi

echo "[submit-exp11] revision=${HEAD} run_id=${RUN_ID} attempt=${ATTEMPT_ID}"
echo "[submit-exp11] 24 Workers x 4 L40S; 3 arms x 32 GPUs; 1000 updates; eval then auto-release"
TMP_YAML="$(mktemp "${TMPDIR:-/tmp}/jexp11-orca24.XXXXXX")"
trap 'rm -f "$TMP_YAML"' EXIT
sed \
  -e "s/EXP11_RUN_ID=unset/EXP11_RUN_ID=${RUN_ID}/" \
  -e "s/EXP11_ATTEMPT_ID=unset/EXP11_ATTEMPT_ID=${ATTEMPT_ID}/" \
  -e "s/EXP11_RESUME=0/EXP11_RESUME=${RESUME}/" \
  "$JOB_TEMPLATE" > "$TMP_YAML"
if [[ "$DRY_RUN" == 1 ]]; then cat "$TMP_YAML"; exit 0; fi
[[ -x "$VTRAINING" ]] || die "vtraining CLI unavailable: $VTRAINING"
"$VTRAINING" run -f "$TMP_YAML"
