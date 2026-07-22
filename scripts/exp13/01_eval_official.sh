#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
SOURCE_RESULTS="${EXP12_SOURCE_RESULTS:-${BASE}/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
ROOT="${EXP13_OFFICIAL_ROOT:-${BASE}/runs/exp13-official/official-budget}"
MAX_CLIPS="${EXP13_OFFICIAL_MAX_CLIPS:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PARTITION_INDEX="${OFFICIAL_PARTITION_INDEX:-0}"
PARTITION_COUNT="${OFFICIAL_PARTITION_COUNT:-1}"
ATTN_IMPLEMENTATION="${EXP13_OFFICIAL_ATTN:-flash_attention_2}"
HEAD="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
OVERLAY="$ROOT/a4_ce_k64_native_overlay.pt"

protocols=(
  official_budget_base_full_generation
  official_budget_ckpt_k64_full_generation
  official_budget_base_cap32_generation
  official_budget_ckpt_k64_cap32_generation
)
overlays=(- "$OVERLAY" - "$OVERLAY")
max_frames=(2048 2048 32 32)
[[ "$PARTITION_COUNT" -eq 4 && "$PARTITION_INDEX" -ge 0 && "$PARTITION_INDEX" -lt 4 ]] || {
  echo "official queue requires partition index 0..3/count=4" >&2; exit 2;
}
protocol="${protocols[$PARTITION_INDEX]}"
overlay="${overlays[$PARTITION_INDEX]}"
frame_cap="${max_frames[$PARTITION_INDEX]}"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${JEPA_ENV}/bin/python"
mkdir -p "$ROOT/logs" "$ROOT/shards"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SHARDS="${#GPUS[@]}"
[[ "$SHARDS" -eq 4 ]] || { echo "official evaluation expects four local GPU shards" >&2; exit 2; }

merged="$ROOT/${protocol}_mvbench.json"
is_complete() {
  "$PY" - "$merged" "$protocol" "$frame_cap" "$MAX_CLIPS" "$HEAD" <<'PY' >/dev/null 2>&1
import json, os, sys
path, protocol, cap, max_clips, commit = sys.argv[1:]
if not os.path.isfile(path): raise SystemExit(1)
d = json.load(open(path)); shards = d.get("metadata", {}).get("shards", [])
ok = (
    d.get("total", 0) > 0 and d.get("protocol") == protocol and len(shards) == 4
    and {int(x.get("max_frames", -1)) for x in shards} == {int(cap)}
    and {int(x.get("max_clips", -1)) for x in shards} == {int(max_clips)}
    and {x.get("frame_policy") for x in shards} == {"official_2fps"}
    and {x.get("prompt_style") for x in shards} == {"official_mvbench"}
    and {x.get("evaluator_commit") for x in shards} == {commit}
)
raise SystemExit(0 if ok else 1)
PY
}
if is_complete; then
  echo "[exp13-official] reuse $merged"
  exit 0
fi

pids=(); shard_files=(); failed=0
for ((shard=0; shard<SHARDS; shard++)); do
  shard_file="$ROOT/shards/${protocol}_mvbench_shard${shard}-of-${SHARDS}.json"
  shard_files+=("$shard_file")
  args=(
    -m jepa_vlm.probes.native_qwen_mcq_eval
    --model "$MODEL_ROOT" --data "$MVB" --task MVBench --output "$shard_file"
    --protocol "$protocol" --frame-policy official_2fps --prompt-style official_mvbench
    --timestamp-fps 2.0 --max-frames "$frame_cap"
    --max-total-video-tokens 224000 --max-tokens-per-unit 640
    --max-clips "$MAX_CLIPS" --num-shards "$SHARDS" --shard-index "$shard"
    --attn-implementation "$ATTN_IMPLEMENTATION"
  )
  [[ "$overlay" == - ]] || args+=(--overlay "$overlay")
  echo "[exp13-official] gpu=${GPUS[$shard]} protocol=$protocol shard=$shard/$SHARDS cap=$frame_cap"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "${args[@]}" \
    > "$ROOT/logs/${protocol}_mvbench_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "official evaluation failed: $protocol" >&2; exit 1; }
"$PY" -m jepa_vlm.probes.merge_mcq_results \
  --inputs "${shard_files[@]}" --output "$merged" \
  > "$ROOT/logs/${protocol}_mvbench_merge.log" 2>&1
is_complete || { echo "merged result failed provenance check: $merged" >&2; exit 1; }
echo "[exp13-official] PASS protocol=$protocol output=$merged"
