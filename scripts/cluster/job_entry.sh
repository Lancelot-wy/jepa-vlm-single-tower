#!/usr/bin/env bash
set -euo pipefail

# Per-pod entry point for the vivolm platform (vtraining / k8s) multi-node job.
# The platform starts THIS script in every Worker pod simultaneously and injects
# TF_CONFIG; env.cluster.sh turns it into torchrun rendezvous args, then
# train_multinode.sh launches torchrun. All pods rendezvous on worker[0]:MASTER_PORT.
# No ssh / mpirun / hostfile needed.
#
# Debug mode: set JOB_SLEEP=1 in the job command to only set up env + sleep, then
# exec into a pod and run scripts by hand.

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

echo "[job_entry] host=$(hostname) NODE_RANK=${NODE_RANK} NNODES=${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[job_entry] gpus: $(nvidia-smi -L 2>/dev/null | grep -c GPU || echo '?')"

if [[ "${JOB_SLEEP:-0}" == "1" ]]; then
  echo "[job_entry] JOB_SLEEP=1 -> env ready, sleeping. Exec in and run:"
  echo "[job_entry]   source scripts/cluster/env.cluster.sh && bash scripts/cluster/train_multinode.sh"
  sleep infinity
fi

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date '+%Y%m%d-%H%M%S')_rank${NODE_RANK}.log"
echo "[job_entry] logging to $LOG_FILE"
set -o pipefail
bash scripts/cluster/train_multinode.sh 2>&1 | tee "$LOG_FILE"
