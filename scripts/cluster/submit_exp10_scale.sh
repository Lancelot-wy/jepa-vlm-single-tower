#!/usr/bin/env bash
# Submit the 128-GPU, 32-Worker EXP-10 job.
# Each arm receives an independent 8-node / 32-GPU DDP group; effective batch
# remains 128 through GRAD_ACCUM=1. The job exits after integrated evaluation.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
JOB_TEMPLATE="${PROJECT_ROOT}/job_exp10_scale.yaml"
RESUME=0
DRY_RUN=0
RUN_ID=""
RUN_ID_EXPLICIT=0

die() { echo "[submit-exp10-scale] ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --run-id)
      shift
      [[ $# -gt 0 ]] || die "--run-id requires a value"
      RUN_ID="$1"
      RUN_ID_EXPLICIT=1
      ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *) die "usage: $0 [--dry-run] [--resume --run-id <existing-run-id>]" ;;
  esac
  shift
done

[[ -d "$PROJECT_ROOT/.git" ]] || die "expected a Git checkout at $PROJECT_ROOT"
[[ -f "$JOB_TEMPLATE" ]] || die "missing scale job template: $JOB_TEMPLATE"
[[ -x "$PROJECT_ROOT/scripts/cluster/job_exp10_scale_entry.sh" ]] || die "missing scale entrypoint"
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || die "checkout has local changes; use a clean committed checkout"

HEAD="$(git rev-parse --short HEAD)"
if [[ -z "$RUN_ID" ]]; then RUN_ID="exp10-scale-$(date '+%Y%m%d-%H%M%S')-${HEAD}"; fi
ATTEMPT_ID="attempt-$(date '+%Y%m%d-%H%M%S')"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "invalid --run-id: $RUN_ID"
if [[ "$RESUME" == "1" && "$RUN_ID_EXPLICIT" != "1" ]]; then
  die "--resume requires --run-id <existing-run-id>"
fi

echo "[submit-exp10-scale] revision=$HEAD run_id=$RUN_ID attempt_id=$ATTEMPT_ID"
echo "[submit-exp10-scale] resources: 32 Workers x 4 L40S = 128 GPUs; 8 nodes / 32 GPUs per arm; effective batch 128"
TMP_YAML="$(mktemp "${TMPDIR:-/tmp}/jexp10-scale.XXXXXX")"
trap 'rm -f "$TMP_YAML"' EXIT
sed \
  -e "s/EXP10_RUN_ID=unset/EXP10_RUN_ID=${RUN_ID}/" \
  -e "s/EXP10_ATTEMPT_ID=unset/EXP10_ATTEMPT_ID=${ATTEMPT_ID}/" \
  -e "s/EXP10_RESUME=0/EXP10_RESUME=${RESUME}/" \
  "$JOB_TEMPLATE" > "$TMP_YAML"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[submit-exp10-scale] dry run; would execute: $VTRAINING run -f $TMP_YAML"
  cat "$TMP_YAML"
  exit 0
fi
[[ -x "$VTRAINING" ]] || die "vtraining CLI is not executable: $VTRAINING"
"$VTRAINING" run -f "$TMP_YAML"
