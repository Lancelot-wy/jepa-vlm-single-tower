#!/usr/bin/env bash
# Submit the canonical full EXP-10 job from a CPU-only vivolm development host.
#
# Usage:
#   bash scripts/cluster/submit_exp10.sh --dry-run
#   bash scripts/cluster/submit_exp10.sh
#   bash scripts/cluster/submit_exp10.sh --resume --run-id <existing-run-id>
#
# The scheduled job is four Workers x four L40S GPUs.  Rank 0 runs audit ->
# prep -> smoke; all four paired arms then train one-per-Worker in parallel;
# rank 0 evaluates.  It is not the historical multi-node Phase-A launcher.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
VTRAINING="${VTRAINING:-/data/vtraining_04/code/vtraining/cli/vtraining}"
JOB_TEMPLATE="${PROJECT_ROOT}/job_exp10.yaml"
RESUME=0
DRY_RUN=0
RUN_ID=""
RUN_ID_EXPLICIT=0
ATTEMPT_ID=""

die() { echo "[submit-exp10] ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --resume) RESUME=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --run-id)
      shift
      [[ $# -gt 0 ]] || die "--run-id requires a value"
      RUN_ID="$1"
      RUN_ID_EXPLICIT=1
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) die "usage: $0 [--dry-run] [--resume]" ;;
  esac
  shift
done

[[ -d "$PROJECT_ROOT/.git" ]] || die "expected a Git checkout at $PROJECT_ROOT; clone/pull the public repository first"
[[ -f "$JOB_TEMPLATE" ]] || die "missing job template: $JOB_TEMPLATE"
[[ -x "$PROJECT_ROOT/scripts/cluster/job_exp10_entry.sh" ]] || die "missing executable EXP-10 job entry"

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  die "checkout has local changes; commit/stash them or use a clean clone so the queued job is reproducible"
fi

HEAD="$(git rev-parse --short HEAD)"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="exp10-$(date '+%Y%m%d-%H%M%S')-${HEAD}"
fi
ATTEMPT_ID="attempt-$(date '+%Y%m%d-%H%M%S')"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "invalid --run-id: $RUN_ID"
if [[ "$RESUME" == "1" && "$RUN_ID_EXPLICIT" != "1" ]]; then
  die "--resume requires --run-id <existing-run-id>; refusing to resume a new output root"
fi
echo "[submit-exp10] repository=$PROJECT_ROOT revision=$HEAD"
echo "[submit-exp10] run_id=$RUN_ID"
echo "[submit-exp10] attempt_id=$ATTEMPT_ID"
echo "[submit-exp10] resources: 4 Worker pods x 4 L40S GPUs = 16 GPUs (one arm per pod)"

# Keep the committed YAML as the exact, reviewable default.  Only the explicit
# --resume flag changes the entry environment; it cannot bypass audit or smoke.
# macOS/BSD `mktemp` substitutes Xs only when they terminate the template;
# keep the temporary job suffix-free so concurrent submitters never collide.
TMP_YAML="$(mktemp "${TMPDIR:-/tmp}/jexp10-curated.XXXXXX")"
trap 'rm -f "$TMP_YAML"' EXIT
sed \
  -e "s/EXP10_RUN_ID=unset/EXP10_RUN_ID=${RUN_ID}/" \
  -e "s/EXP10_ATTEMPT_ID=unset/EXP10_ATTEMPT_ID=${ATTEMPT_ID}/" \
  -e "s/EXP10_RESUME=0/EXP10_RESUME=${RESUME}/" \
  "$JOB_TEMPLATE" > "$TMP_YAML"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[submit-exp10] dry run; would execute: $VTRAINING run -f $TMP_YAML"
  echo "----- rendered job -----"
  cat "$TMP_YAML"
  exit 0
fi

[[ -x "$VTRAINING" ]] || die "vtraining CLI is not executable: $VTRAINING"
echo "[submit-exp10] submitting full EXP-10 (resume=${RESUME})"
"$VTRAINING" run -f "$TMP_YAML"
