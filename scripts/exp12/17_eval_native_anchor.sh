#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
SOURCE_RESULTS="${EXP12_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ROOT="${EXP12_NATIVE_ANCHOR_ROOT:-${BASE}/runs/exp13/native-anchor}"
MAX_CLIPS="${MAX_CLIPS:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
FORCE="${FORCE:-0}"
NUM_FRAMES="${NUM_FRAMES:-32}"
TIMESTAMP_FPS="${TIMESTAMP_FPS:-4.0}"
PARTITION_INDEX="${NATIVE_PARTITION_INDEX:-0}"
PARTITION_COUNT="${NATIVE_PARTITION_COUNT:-1}"

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${JEPA_ENV}/bin/python"
HEAD="$(git rev-parse HEAD)"
mkdir -p "$ROOT/logs" "$ROOT/shards"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SHARDS="${SHARDS:-${#GPUS[@]}}"
[[ "$SHARDS" -ge 1 && "$SHARDS" -le "${#GPUS[@]}" ]] || {
  echo "SHARDS must be between 1 and the number of local GPUs" >&2; exit 2;
}
[[ "$PARTITION_COUNT" -ge 1 && "$PARTITION_INDEX" -ge 0 && "$PARTITION_INDEX" -lt "$PARTITION_COUNT" ]] || {
  echo "invalid NATIVE_PARTITION_INDEX/COUNT" >&2; exit 2;
}

is_complete() {
  "$PY" - "$1" "$2" "$MAX_CLIPS" "$HEAD" <<'PY' >/dev/null 2>&1
import json, os, sys
path, protocol, max_clips, commit = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if not os.path.isfile(path): raise SystemExit(1)
d=json.load(open(path)); shards=d.get("metadata", {}).get("shards", [])
seen={int(x.get("max_clips", -1)) for x in shards}
commits={x.get("evaluator_commit") for x in shards}
raise SystemExit(0 if d.get("total", 0)>0 and d.get("protocol")==protocol and seen=={max_clips} and commits=={commit} else 1)
PY
}

OVERLAY="$ROOT/a4_ce_k64_native_overlay.pt"
EXPECTED_CHECKPOINT="$SOURCE_RESULTS/a4_ce_k64/checkpoint-800/state.pt"
overlay_valid() {
  "$PY" - "$OVERLAY" "$EXPECTED_CHECKPOINT" <<'PY' >/dev/null 2>&1
import json, os, sys
overlay, expected = sys.argv[1:]
metadata = json.load(open(overlay + ".json"))
raise SystemExit(0 if os.path.abspath(metadata.get("source_checkpoint", "")) == os.path.abspath(expected) else 1)
PY
}
if [[ "$FORCE" == 1 || ! -s "$OVERLAY" ]] || ! overlay_valid; then
  echo "[native-anchor] exporting model-only checkpoint overlay once"
  "$PY" -m jepa_vlm.probes.native_checkpoint \
    --checkpoint "$EXPECTED_CHECKPOINT" --output "$OVERLAY" \
    > "$ROOT/logs/export_native_overlay.log" 2>&1
fi

run_protocol_task() {
  protocol="$1"; overlay="$2"; task="$3"; data="$4"; slug="${task,,}"
  merged="$ROOT/${protocol}_${slug}.json"
  if [[ "$FORCE" != 1 ]] && is_complete "$merged" "$protocol"; then
    echo "[native-anchor] reuse $merged"
    return 0
  fi
  pids=(); shard_files=(); failed=0
  for ((shard=0; shard<SHARDS; shard++)); do
    shard_file="$ROOT/shards/${protocol}_${slug}_shard${shard}-of-${SHARDS}.json"
    shard_files+=("$shard_file")
    args=(
      -m jepa_vlm.probes.native_qwen_mcq_eval
      --model "$MODEL_ROOT" --data "$data" --task "$task" --output "$shard_file"
      --protocol "$protocol" --num-frames "$NUM_FRAMES" --timestamp-fps "$TIMESTAMP_FPS"
      --max-clips "$MAX_CLIPS" --num-shards "$SHARDS" --shard-index "$shard"
    )
    [[ "$overlay" == - ]] || args+=(--overlay "$overlay")
    echo "[native-anchor] gpu=${GPUS[$shard]} protocol=$protocol task=$task shard=$shard/$SHARDS"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "${args[@]}" \
      > "$ROOT/logs/${protocol}_${slug}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]] || return 1
  "$PY" -m jepa_vlm.probes.merge_mcq_results \
    --inputs "${shard_files[@]}" --output "$merged" \
    > "$ROOT/logs/${protocol}_${slug}_merge.log" 2>&1
}

protocols=(
  "native_base_matched32_generation|-"
  "native_ckpt_k64_matched32_generation|$OVERLAY"
)
tasks=("MVBench|$MVB" "Tempcompass|$TC")
combination=0
for spec in "${protocols[@]}"; do
  IFS='|' read -r protocol overlay <<< "$spec"
  for task_spec in "${tasks[@]}"; do
    IFS='|' read -r task data <<< "$task_spec"
    if (( combination % PARTITION_COUNT != PARTITION_INDEX )); then
      combination=$((combination + 1))
      continue
    fi
    combination=$((combination + 1))
    run_protocol_task "$protocol" "$overlay" "$task" "$data" || {
      echo "[native-anchor] failed protocol=$protocol task=$task" >&2; exit 1;
    }
  done
done
echo "[native-anchor] PASS root=$ROOT"
