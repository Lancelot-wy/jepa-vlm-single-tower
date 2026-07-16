#!/usr/bin/env bash
set -euo pipefail

# Batch-submit jepa-vlm-single-tower cluster experiments to the vivolm platform.
# Each experiment == one 2-node x 4-GPU job running a configs/cl_<name>.yaml variant
# (all inherit vivolm_llava_video.yaml -> same LLaVA-Video data / local model / sdpa,
#  differing only in one ablation knob + a unique output_dir).
#
# Usage:
#   scripts/cluster/submit_batch.sh v1 v22 mask25 mask75 mtp_off mtp_k1 bidir frozen_vit patch_mask lora
#   scripts/cluster/submit_batch.sh --dry-run mask75 mtp_off        # print yamls, don't submit
#
# Notes:
#  * The main run (v21) uses configs/vivolm_llava_video.yaml and is submitted via job.yaml.
#  * Each job gets a unique type/name so logs & outputs never collide.

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"

VTRAINING="/data/vtraining_04/code/vtraining/cli/vtraining"
IMAGE="registry-wl01.vivo.lan/romai_dev/images/llava_train:vivolm-ngc-25.10-2604091110"
BUSINESS="${BUSINESS:-VideoFoundationModel1b-wl01}"
NODES="${NODES:-2}"   # V4 validation: NODES=1 => one experiment per 4xL40S node

DRY_RUN=0
EXPS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else EXPS+=("$a"); fi
done
if [[ ${#EXPS[@]} -eq 0 ]]; then
  echo "usage: $0 [--dry-run] <exp> [<exp> ...]   (exp = suffix of configs/cl_<exp>.yaml)" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for name in "${EXPS[@]}"; do
  # accept either configs/<name>.yaml (e.g. r2_v21) or the legacy configs/cl_<name>.yaml
  cfg="configs/${name}.yaml"
  [[ -f "$cfg" ]] || cfg="configs/cl_${name}.yaml"
  if [[ ! -f "$cfg" ]]; then echo "!! missing configs/${name}.yaml and $cfg -- skipping" >&2; continue; fi
  yaml="$TMPDIR/job_${name}.yaml"
  # platform `type` allows only [alpha][num]- ; map underscores to dashes.
  jobtype="jepa-vlm-${name//_/-}"
  # optional per-batch overrides (e.g. EXTRA_OVERRIDES='train.min_flow=8.42')
  cmd="CONFIG=${cfg}"
  [[ -n "${EXTRA_OVERRIDES:-}" ]] && cmd="${cmd} EXTRA_OVERRIDES='${EXTRA_OVERRIDES}'"
  cmd="${cmd} bash ${PROJECT_ROOT}/scripts/cluster/job_entry.sh"
  cat > "$yaml" <<EOF
type: ${jobtype}
business: ${BUSINESS}
image: ${IMAGE}
dataPaths:
- /data/vjuicefs_sz_ocr_wl/public_data
- /data/vjuicefs_ai_ocr_wl/public_data
- /data/vjuicefs_ai_gpt_vision_wl04/public_data
tmpfs: true
restartPolicy: Never
run:
  rdma: 'ib'
  command: "${cmd}"
spec:
  Worker:
    num: ${NODES}
    nodes:
      vivo.com/machine-type: 'L40s'
    limits:
      gpu: "4"
      cpu: "120"
      memory: "990Gi"
    requests:
      gpu: "4"
      cpu: "120"
      memory: "990Gi"
EOF
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== $yaml ====="; cat "$yaml"; echo
  else
    echo ">> submitting $name ($cfg) as $jobtype"
    "$VTRAINING" run -f "$yaml" || echo "!! submit FAILED for $name -- continuing" >&2
  fi
done
