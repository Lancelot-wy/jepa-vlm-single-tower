#!/usr/bin/env bash
# Cluster env for jepa-vlm-single-tower on the vivolm platform (L40s, 4 GPU/node).
# Usage (sourced by job_entry.sh, or manually in a debug pod):
#   source scripts/cluster/env.cluster.sh
#
# Everything lives on the shared /data (juicefs) mount, so paths are identical on
# every node. We reuse the already-built conda env envs/jepa311 (python 3.11 +
# torch 2.5.1+cu124 + transformers 5.6.0 + accelerate 1.11 + av 16 + peft 0.18),
# so NO pip install and NO custom image build are needed.

CLUSTER_BASE="/data/vjuicefs_sz_ocr_wl/public_data/11193960"
PROJECT_ROOT="${CLUSTER_BASE}/jepa-vlm-single-tower"

# ---- python env (shared conda env on /data; offline nodes use it directly) ----
# The container's stock LD_LIBRARY_PATH points at the old torch 1.13 libs and
# conflicts with torch 2.5, so it MUST be cleared.
export JEPA_ENV="${CLUSTER_BASE}/envs/jepa311"
export PATH="${JEPA_ENV}/bin:${PATH}"
# The cluster python does not add the CWD to sys.path, and `python scripts/x.py`
# only puts scripts/ on the path. Put the repo root on PYTHONPATH so every python
# invocation (preflight, train, eval, probes) can `import jepa_vlm`.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

# ---- model / outputs (also referenced by configs/vivolm_ssv2.yaml) ----
export QWEN3_VL_MODEL_PATH="${CLUSTER_BASE}/models/Qwen3-VL-2B-Instruct"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${CLUSTER_BASE}/outputs}"

# ---- distributed launch knobs ----
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c GPU || echo 4)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -n "${TF_CONFIG:-}" ]]; then
  # The platform starts the SAME command in every Worker pod and injects TF_CONFIG:
  #   {"cluster":{"worker":["10.x.a:2222",...]},"task":{"type":"worker","index":N}}
  # Derive torchrun rendezvous from it (same logic vivolm uses); pods rendezvous on
  # worker[0]:MASTER_PORT. No hard-coded IPs, no ssh/mpirun/hostfile.
  _tf_workers=$("${JEPA_ENV}/bin/python" - <<'PY'
import json, os
cfg = json.loads(os.environ["TF_CONFIG"])
print(",".join(cfg["cluster"]["worker"]))
PY
)
  _tf_index=$("${JEPA_ENV}/bin/python" - <<'PY'
import json, os
cfg = json.loads(os.environ["TF_CONFIG"])
print(cfg["task"]["index"])
PY
)
  _tf_workers=$(echo "${_tf_workers}" | sed 's/:[0-9]\+//g')   # strip ":sshport"
  IFS=',' read -r -a _workers <<< "${_tf_workers}"
  export MASTER_ADDR="${_workers[0]}"
  export NNODES="${#_workers[@]}"
  export NODE_RANK="${_tf_index}"
  echo "[env.cluster] TF_CONFIG detected: workers=(${_workers[*]})"
else
  # Manual / single-node fallback (debugging in one pod). Override as needed:
  #   export NNODES=3 NODE_RANK=0 MASTER_ADDR=<worker0_ip>
  export NNODES="${NNODES:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  echo "[env.cluster] no TF_CONFIG; manual NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}"
fi

# ---- HF / misc: data + weights are fully local, forbid any network hit ----
export HF_HOME="${OUTPUT_ROOT}/hf_cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# Perf / stability knobs mirrored from vivolm example scripts.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[env.cluster] NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
