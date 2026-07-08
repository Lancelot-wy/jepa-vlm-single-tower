#!/usr/bin/env bash
set -uo pipefail

# MVBench / TempCompass multiple-choice eval (jepa_vlm.probes.mcq_eval) as a
# single-GPU platform job. For the given arm + ckpt it runs each task and appends
# the summary to mcq_eval_results.txt. Choices are scored by answer likelihood
# (the single-tower model isn't vLLM-servable); see mcq_eval.py.
#
#   ONLY_ARM=r3_joint bash scripts/cluster/run_mcq_eval.sh
# Env: STEP (default 4000), TASKS (default "MVBench Tempcompass"),
#      MAXCLIPS (default 0 = all), DATA (merged offline-eval jsonl).

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUTPUT_ROOT}"
DATA="${DATA:-/data/vjuicefs_ai_nlp_wl/public_data/EVAL_DATA/vllm_data/mllm_offline/2026/merge_all_0306_sample_data.jsonl}"
STEP="${STEP:-4000}"
TASKS="${TASKS:-MVBench Tempcompass}"
MAXCLIPS="${MAXCLIPS:-0}"
# Smoke runs (max_clips>0) go to a throwaway file so partial numbers never
# satisfy the full run's skip-if-done grep on mcq_eval_results.txt.
if [[ "$MAXCLIPS" == "0" ]]; then RESULTS="$OUT/mcq_eval_results.txt"; else RESULTS="$OUT/mcq_eval_smoke.txt"; fi
echo "# MCQ eval  $(date)  (step=$STEP max_clips=$MAXCLIPS)" >> "$RESULTS"

cap=()
[[ "$MAXCLIPS" != "0" ]] && cap=(--max-clips "$MAXCLIPS")

for arm in r3_joint r3_sft; do
  if [[ -n "${ONLY_ARM:-}" && "$ONLY_ARM" != "$arm" ]]; then continue; fi
  cfg="$OUT/$arm/config.json"
  ck="$OUT/$arm/step_${STEP}"
  [[ -f "$cfg" ]] || { echo "!! $arm: missing $cfg -- skip" | tee -a "$RESULTS"; continue; }
  [[ -d "$ck" ]]  || { echo "!! $arm: missing $ck -- skip"  | tee -a "$RESULTS"; continue; }
  for task in $TASKS; do
    tag="$arm step_${STEP} $task"
    if [[ "$MAXCLIPS" == "0" ]] && grep -q "^${tag}:" "$RESULTS" 2>/dev/null; then
      echo "   $tag done -> skip"; continue
    fi
    echo ">> [$tag] MCQ eval"
    out="$(python -m jepa_vlm.probes.mcq_eval --config "$cfg" --ckpt "$ck" \
      --data "$DATA" --task "$task" "${cap[@]}" 2>&1)" \
      || { echo "!! $tag: FAILED" | tee -a "$RESULTS"; echo "$out" | tail -8; continue; }
    ov="$(echo "$out" | grep -E '^overall:' | head -1 | sed 's/^overall: *//')"
    echo "$tag: overall=$ov" | tee -a "$RESULTS"
    echo "$out" | grep -E '^  ' | sed "s/^/    [$task] /" >> "$RESULTS"
  done
done

echo "=== MCQ eval done -> $RESULTS ==="
grep -E "^r3_" "$RESULTS" | tail -20
