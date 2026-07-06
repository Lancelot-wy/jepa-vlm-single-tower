#!/usr/bin/env bash
set -euo pipefail

# Submit the temporal linear-probe matrix (scripts/cluster/run_probes.sh) as
# PARALLEL single-node / single-GPU vivolm jobs -- one job per model so the whole
# 6-model matrix finishes in ~one model's wall-clock instead of serially. Each job
# sets ONLY_MODEL so it handles just its model; run_probes.sh skips any feature
# file that already exists (safe to resubmit). Run AFTER the round-2 arms produced
# step_2000 checkpoints.
#   scripts/cluster/submit_probes.sh                       # all 6 models
#   scripts/cluster/submit_probes.sh r2_v21 r2_residual    # subset
#   scripts/cluster/submit_probes.sh --dry-run r2_v21      # print yaml only

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"

VTRAINING="/data/vtraining_04/code/vtraining/cli/vtraining"
IMAGE="registry-wl01.vivo.lan/romai_dev/images/llava_train:vivolm-ngc-25.10-2604091110"
BUSINESS="VideoFoundationModel1b-wl01"

DRY_RUN=0
MODELS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else MODELS+=("$a"); fi
done
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(base r2_frozen r2_v21 r2_varreg r2_residual r2_sft_baseline)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for m in "${MODELS[@]}"; do
  # platform caps `type` at 32 chars; keep the prefix short.
  jobtype="jr2p-${m//_/-}"
  yaml="$TMPDIR/${m}.yaml"
  cat > "$yaml" <<EOF
type: ${jobtype}
business: ${BUSINESS}
image: ${IMAGE}
dataPaths:
- /data/vjuicefs_sz_ocr_wl/public_data
- /data/vjuicefs_ai_ocr_wl/public_data
tmpfs: true
restartPolicy: Never
run:
  rdma: 'ib'
  command: "ONLY_MODEL=${m} bash ${PROJECT_ROOT}/scripts/cluster/run_probes.sh"
spec:
  Worker:
    num: 1
    nodes:
      vivo.com/machine-type: 'L40s'
    limits:
      gpu: "1"
      cpu: "30"
      memory: "200Gi"
    requests:
      gpu: "1"
      cpu: "30"
      memory: "200Gi"
EOF
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== $yaml ====="; cat "$yaml"; echo
  else
    echo ">> submitting probe job for $m ($jobtype)"
    "$VTRAINING" run -f "$yaml" || echo "!! submit FAILED for $m" >&2
  fi
done
