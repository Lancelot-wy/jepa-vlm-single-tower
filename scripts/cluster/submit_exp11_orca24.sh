#!/usr/bin/env bash
# Submit/resume the 24-Worker EXP-11 overnight pilot.
# Usage:
#   submit_exp11_orca24.sh --control-dir /path/to/exp11_frozen_sft_s0 \
#     [--control-world-size 32] [--dry-run]
#   submit_exp11_orca24.sh --resume --run-id <id> --control-dir /same/path

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
JOB_TEMPLATE="${PROJECT_ROOT}/job_exp11_orca24.yaml"
RESUME=0
DRY_RUN=0
RUN_ID=""
RUN_ID_EXPLICIT=0
CONTROL_DIR="${EXP11_CONTROL_DIR:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/outputs/exp11_frozen_sft_s0}"
CONTROL_WORLD_SIZE="${EXP11_CONTROL_WORLD_SIZE:-32}"

die() { echo "[submit-exp11] ERROR: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --run-id) shift; [[ $# -gt 0 ]] || die "--run-id requires a value"; RUN_ID="$1"; RUN_ID_EXPLICIT=1 ;;
    --control-dir) shift; [[ $# -gt 0 ]] || die "--control-dir requires a value"; CONTROL_DIR="$1" ;;
    --control-world-size) shift; [[ $# -gt 0 ]] || die "--control-world-size requires a value"; CONTROL_WORLD_SIZE="$1" ;;
    -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
    *) die "usage: $0 --control-dir <path> [--control-world-size N] [--dry-run] [--resume --run-id <existing-run-id>]" ;;
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
[[ "$CONTROL_DIR" =~ ^/[A-Za-z0-9_./-]+$ ]] || die "control dir must be a safe absolute path"
[[ "$CONTROL_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || die "control world size must be positive"
if [[ "$RESUME" == 1 && "$RUN_ID_EXPLICIT" != 1 ]]; then die "--resume requires --run-id"; fi

echo "[submit-exp11] revision=${HEAD} run_id=${RUN_ID} attempt=${ATTEMPT_ID}"
echo "[submit-exp11] reused_control=${CONTROL_DIR} declared_control_world=${CONTROL_WORLD_SIZE}"
echo "[submit-exp11] 24 Workers x 4 L40S; 3 new arms x 32 GPUs; 4000 updates; 4-arm eval then auto-release"
TMP_YAML="$(mktemp "${TMPDIR:-/tmp}/jexp11-orca24.XXXXXX")"
trap 'rm -f "$TMP_YAML"' EXIT
sed \
  -e "s/EXP11_RUN_ID=unset/EXP11_RUN_ID=${RUN_ID}/" \
  -e "s/EXP11_ATTEMPT_ID=unset/EXP11_ATTEMPT_ID=${ATTEMPT_ID}/" \
  -e "s/EXP11_RESUME=0/EXP11_RESUME=${RESUME}/" \
  -e "s|EXP11_CONTROL_DIR=unset|EXP11_CONTROL_DIR=${CONTROL_DIR}|" \
  -e "s/EXP11_CONTROL_WORLD_SIZE=32/EXP11_CONTROL_WORLD_SIZE=${CONTROL_WORLD_SIZE}/" \
  "$JOB_TEMPLATE" > "$TMP_YAML"
if [[ "$DRY_RUN" == 1 ]]; then cat "$TMP_YAML"; exit 0; fi
[[ -x "$VTRAINING" ]] || die "vtraining CLI unavailable: $VTRAINING"
"$VTRAINING" run -f "$TMP_YAML"
