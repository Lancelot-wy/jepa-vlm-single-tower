#!/usr/bin/env bash
set -euo pipefail

# Multi-node Phase A / Phase B training for jepa-vlm-single-tower.
#
# This is the vivolm-harness equivalent of the repo's `accelerate launch`: it swaps
# the launcher only (README section 6). jepa_vlm.train uses accelerate's Accelerator(),
# which reads the standard torchrun env (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/PORT),
# so `torchrun -m jepa_vlm.train` needs no code change.
#
# Run on EACH pod (job_entry.sh does this); env.cluster.sh has already exported
# NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / NPROC_PER_NODE from TF_CONFIG.
#
# Overridable env:
#   CONFIG=configs/vivolm_ssv2.yaml   # which config to run
#   GRAD_ACCUM=4                      # per-device grad-accum (eff.batch = bs*accum*world)
#   EXTRA_OVERRIDES="model.mask_ratio=0.75 train.max_steps=8000"   # any key=value

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

CONFIG="${CONFIG:-configs/vivolm_ssv2.yaml}"

OVERRIDES=()
[[ -n "${GRAD_ACCUM:-}" ]] && OVERRIDES+=("train.grad_accum=${GRAD_ACCUM}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_OVERRIDES:-}" ]] && OVERRIDES+=(${EXTRA_OVERRIDES})

DISTRIBUTED_LAUNCH_ARGS=(
  --nproc_per_node "${NPROC_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

echo "[train] config=${CONFIG} world=$((NPROC_PER_NODE*NNODES)) overrides=(${OVERRIDES[*]:-none})"
set -x
torchrun "${DISTRIBUTED_LAUNCH_ARGS[@]}" \
  -m jepa_vlm.train --config "${CONFIG}" "${OVERRIDES[@]}"
