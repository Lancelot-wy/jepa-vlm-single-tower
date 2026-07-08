#!/usr/bin/env bash
set -uo pipefail

# Round-3 held-out temporal-QA evaluation (EXPERIMENTS.md Round-3 readout #2) as a
# single-GPU platform job. For each arm (r3_joint / r3_sft) and each saved ckpt
# (step_1000..4000) it runs jepa_vlm.probes.temporal_qa_eval on the SAME val
# manifest + seed, appending "arm step: overall/shuffle/reverse" to qa_eval_results.txt.
#
# Local model loads exceed the tool timeout and can't background, so eval runs here.
# ONLY_ARM lets one job handle one arm so the two arms run in parallel.
#   ONLY_ARM=r3_joint bash scripts/cluster/run_qa_eval.sh
# Overridable env: MAXCLIPS (default 500), SEED (default 0), STEPS (default "1000 2000 3000 4000").

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data"
OUT="${OUTPUT_ROOT}"
VAL="$DATA/llava_video/val.jsonl"
MAXCLIPS="${MAXCLIPS:-500}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1000 2000 3000 4000}"
RESULTS="$OUT/qa_eval_results.txt"
echo "# temporal QA eval  $(date)  (max_clips=$MAXCLIPS seed=$SEED)" >> "$RESULTS"

for arm in r3_joint r3_sft; do
  if [[ -n "${ONLY_ARM:-}" && "$ONLY_ARM" != "$arm" ]]; then continue; fi
  cfg="$OUT/$arm/config.json"
  if [[ ! -f "$cfg" ]]; then echo "!! $arm: missing $cfg -- skip" | tee -a "$RESULTS"; continue; fi
  for st in $STEPS; do
    ck="$OUT/$arm/step_${st}"
    [[ -d "$ck" ]] || { echo "!! $arm step_$st: missing ckpt -- skip" | tee -a "$RESULTS"; continue; }
    if grep -q "^${arm} step_${st}:" "$RESULTS" 2>/dev/null; then echo "   $arm step_$st done -> skip"; continue; fi
    echo ">> [$arm/step_$st] temporal QA eval"
    out="$(python -m jepa_vlm.probes.temporal_qa_eval \
      --config "$cfg" --ckpt "$ck" --manifest "$VAL" \
      --max-clips "$MAXCLIPS" --seed "$SEED" 2>&1)" || { echo "!! $arm step_$st: eval FAILED" | tee -a "$RESULTS"; echo "$out" | tail -5; continue; }
    ov="$(echo "$out"  | grep -E '^overall:'   | head -1 | sed 's/^overall: *//')"
    sh="$(echo "$out"  | grep -E '^  shuffle'  | head -1 | sed 's/^ *//')"
    rv="$(echo "$out"  | grep -E '^  reverse'  | head -1 | sed 's/^ *//')"
    nn="$(echo "$out"  | grep -E '^  none'     | head -1 | sed 's/^ *//')"
    echo "$arm step_$st: overall=$ov | $nn | $sh | $rv" | tee -a "$RESULTS"
  done
done

echo "=== qa eval done -> $RESULTS ==="
grep -E "^r3_" "$RESULTS" | tail -20
