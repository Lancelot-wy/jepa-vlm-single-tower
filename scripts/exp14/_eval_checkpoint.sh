#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: _eval_checkpoint.sh 400|800 <arm>}"
ARM="${2:?usage: _eval_checkpoint.sh 400|800 <arm>}"
[[ "$STEP" == 400 || "$STEP" == 800 ]] || { echo "step must be 400 or 800" >&2; exit 2; }
case "$ARM" in
  b0_ce_seed1|b1_query_seed1|b2_noquery_seed0|b3_noquery_seed1|\
  b4_query_beatcopy_seed0|b5_query_beatcopy_seed1) ;;
  *) echo "unknown EXP-14 arm: $ARM" >&2; exit 2 ;;
esac
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
RUN_ROOT="${EXP14_RUN_ROOT:?EXP14_RUN_ROOT is required}"
ROOT="${RUN_ROOT}/results/exp14_state_diagnostics"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
mkdir -p "$ROOT/eval"

if [[ "$STEP" == 400 ]]; then
  subset="$ROOT/eval/tempcompass_checkpoint400_subset.jsonl"
  ids="$ROOT/eval/tempcompass_checkpoint400_subset_ids.tsv"
  if [[ ! -s "$subset" ]]; then
    lock="$ROOT/eval/.checkpoint400_subset.lock"
    if mkdir "$lock" 2>/dev/null; then
      trap 'rm -rf "$lock"' EXIT
      "$PY" scripts/exp12/make_tempcompass_subset.py \
        --input "$TC" --output "$lock/subset.jsonl" --ids "$lock/ids.tsv" \
        > "$ROOT/eval/subset_build.log"
      mv "$lock/subset.jsonl" "$subset"
      mv "$lock/ids.tsv" "$ids"
      mv "$lock/subset.jsonl.sha256" "$subset.sha256"
      rm -rf "$lock"; trap - EXIT
    else
      deadline=$(( $(date +%s) + 1800 ))
      while [[ ! -s "$subset" ]]; do
        (( $(date +%s) < deadline )) || { echo "timed out waiting for subset" >&2; exit 1; }
        sleep 5
      done
    fi
  fi
  TASKS=(Tempcompass); DATASETS=("$subset"); SUFFIXES=(tempcompass_subset)
else
  TASKS=(MVBench Tempcompass); DATASETS=("$MVB" "$TC"); SUFFIXES=(mvbench tempcompass)
fi

out="$ROOT/$ARM"
[[ -f "$out/checkpoint-${STEP}/state.pt" ]] || { echo "missing $ARM checkpoint-$STEP" >&2; exit 1; }
for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"; data="${DATASETS[$index]}"; suffix="${SUFFIXES[$index]}"
  CUDA_VISIBLE_DEVICES="${EXP14_EVAL_GPU:-0}" "$PY" -m jepa_vlm.probes.mcq_eval \
    --config "$out/config.json" --ckpt "$out/checkpoint-${STEP}" \
    --data "$data" --task "$task" --output "$out/checkpoint-${STEP}_${suffix}.json" \
    > "$out/checkpoint-${STEP}_${suffix}.eval.log" 2>&1
done
echo "[exp14-eval] PASS arm=$ARM checkpoint-$STEP"
