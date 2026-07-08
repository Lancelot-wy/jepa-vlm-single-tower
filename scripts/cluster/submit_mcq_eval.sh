#!/usr/bin/env bash
set -euo pipefail

# Submit the MVBench / TempCompass MCQ eval (scripts/cluster/run_mcq_eval.sh) as
# PARALLEL single-node / single-GPU vivolm jobs -- one job per arm so both finish
# in ~one arm's wall-clock. Each job sets ONLY_ARM; run_mcq_eval.sh skips arm/step/
# task triples already recorded when MAXCLIPS=0 (safe to resubmit). Run AFTER
# r3_joint/r3_sft finished training.
#   scripts/cluster/submit_mcq_eval.sh                          # both arms, all clips
#   MAXCLIPS=20 scripts/cluster/submit_mcq_eval.sh r3_joint     # smoke subset
#   scripts/cluster/submit_mcq_eval.sh --dry-run
# Env forwarded into the job: STEP, TASKS, MAXCLIPS, DATA (see run_mcq_eval.sh).

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"

VTRAINING="/data/vtraining_04/code/vtraining/cli/vtraining"
IMAGE="registry-wl01.vivo.lan/romai_dev/images/llava_train:vivolm-ngc-25.10-2604091110"
BUSINESS="VideoFoundationModel1b-wl01"

STEP="${STEP:-4000}"
TASKS="${TASKS:-MVBench Tempcompass}"
MAXCLIPS="${MAXCLIPS:-0}"
DATA="${DATA:-/data/vjuicefs_ai_nlp_wl/public_data/EVAL_DATA/vllm_data/mllm_offline/2026/merge_all_0306_sample_data.jsonl}"

DRY_RUN=0
ARMS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY_RUN=1; else ARMS+=("$a"); fi
done
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=(r3_joint r3_sft)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for arm in "${ARMS[@]}"; do
  jobtype="jr3mcq-${arm//_/-}"
  yaml="$TMPDIR/${arm}.yaml"
  cat > "$yaml" <<EOF
type: ${jobtype}
business: ${BUSINESS}
image: ${IMAGE}
dataPaths:
- /data/vjuicefs_sz_ocr_wl/public_data
- /data/vjuicefs_ai_ocr_wl/public_data
- /data/vjuicefs_ai_nlp_wl/public_data
- /data/vjuicefs_ai_gpt_nlp/public_data
tmpfs: true
restartPolicy: Never
run:
  rdma: 'ib'
  command: "STEP=${STEP} TASKS='${TASKS}' MAXCLIPS=${MAXCLIPS} DATA='${DATA}' ONLY_ARM=${arm} bash ${PROJECT_ROOT}/scripts/cluster/run_mcq_eval.sh"
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
    echo ">> submitting MCQ eval for $arm ($jobtype)  step=$STEP max_clips=$MAXCLIPS"
    "$VTRAINING" run -f "$yaml" || echo "!! submit FAILED for $arm" >&2
  fi
done
