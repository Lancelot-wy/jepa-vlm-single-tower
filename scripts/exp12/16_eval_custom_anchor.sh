#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
SOURCE_RESULTS="${EXP12_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ROOT="${EXP12_NATIVE_ANCHOR_ROOT:-${BASE}/runs/exp13/native-anchor}"
MAX_CLIPS="${MAX_CLIPS:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
FORCE="${FORCE:-0}"
PARTITION_INDEX="${CUSTOM_PARTITION_INDEX:-0}"
PARTITION_COUNT="${CUSTOM_PARTITION_COUNT:-1}"

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${JEPA_ENV}/bin/python"
HEAD="$(git rev-parse HEAD)"
mkdir -p "$ROOT/logs"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "empty GPU_LIST" >&2; exit 2; }
[[ "$PARTITION_COUNT" -ge 1 && "$PARTITION_INDEX" -ge 0 && "$PARTITION_INDEX" -lt "$PARTITION_COUNT" ]] || {
  echo "invalid CUSTOM_PARTITION_INDEX/COUNT" >&2; exit 2;
}

is_complete() {
  "$PY" - "$1" "$2" "$MAX_CLIPS" "$HEAD" "$3" <<'PY' >/dev/null 2>&1
import json, os, sys
path, protocol, max_clips, commit, checkpoint = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
if not os.path.isfile(path): raise SystemExit(1)
d=json.load(open(path)); m=d.get("metadata", {})
expected_checkpoint = None if checkpoint == "-" else os.path.abspath(checkpoint)
valid = (
    d.get("total", 0) > 0
    and d.get("protocol") == protocol
    and int(m.get("max_clips", -1)) == max_clips
    and m.get("evaluator_commit") == commit
    and m.get("checkpoint") == expected_checkpoint
)
raise SystemExit(0 if valid else 1)
PY
}

run_one() {
  protocol="$1"; config="$2"; checkpoint="$3"; answer_format="$4"
  task="$5"; data="$6"; gpu="$7"; slug="${task,,}"
  output="$ROOT/${protocol}_${slug}.json"
  if [[ "$FORCE" != 1 ]] && is_complete "$output" "$protocol" "$checkpoint"; then
    echo "[custom-anchor] reuse $output"
    return 0
  fi
  args=(
    -m jepa_vlm.probes.mcq_eval
    --config "$config" --data "$data" --task "$task"
    --max-clips "$MAX_CLIPS" --answer-format "$answer_format"
    --protocol "$protocol" --output "$output"
  )
  [[ "$checkpoint" == - ]] || args+=(--ckpt "$checkpoint")
  echo "[custom-anchor] gpu=$gpu protocol=$protocol task=$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${args[@]}" > "$ROOT/logs/${protocol}_${slug}.log" 2>&1
}

specs=(
  "custom_base_k4_full_option|configs/orca_token_sweep/a0_ce_k4.yaml|-|full_option"
  "custom_base_k16_full_option|configs/orca_token_sweep/a2_ce_k16.yaml|-|full_option"
  "custom_base_k64_full_option|configs/orca_token_sweep/a4_ce_k64.yaml|-|full_option"
  "custom_ckpt_k64_full_option|$SOURCE_RESULTS/a4_ce_k64/config.json|$SOURCE_RESULTS/a4_ce_k64/checkpoint-800|full_option"
  "custom_base_k64_letter|configs/orca_token_sweep/a4_ce_k64.yaml|-|letter"
  "custom_ckpt_k64_letter|$SOURCE_RESULTS/a4_ce_k64/config.json|$SOURCE_RESULTS/a4_ce_k64/checkpoint-800|letter"
)
tasks=("MVBench|$MVB" "Tempcompass|$TC")
pids=(); slot=0; failed=0; combination=0
for spec in "${specs[@]}"; do
  IFS='|' read -r protocol config checkpoint answer_format <<< "$spec"
  for task_spec in "${tasks[@]}"; do
    IFS='|' read -r task data <<< "$task_spec"
    if (( combination % PARTITION_COUNT != PARTITION_INDEX )); then
      combination=$((combination + 1))
      continue
    fi
    combination=$((combination + 1))
    run_one "$protocol" "$config" "$checkpoint" "$answer_format" "$task" "$data" "${GPUS[$slot]}" &
    pids+=("$!"); slot=$((slot + 1))
    if [[ "$slot" -eq "${#GPUS[@]}" ]]; then
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      pids=(); slot=0
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "[custom-anchor] one or more evaluations failed" >&2; exit 1; }
echo "[custom-anchor] PASS root=$ROOT"
