#!/usr/bin/env bash
# Single-node multi-GPU eval of the EXP-10 scale128 step_500 checkpoints.
# Runs the 8 (arm, task) pairs in parallel, one per GPU, then writes a
# scorecard.  Full dataset by default (EVAL_MAXCLIPS=0).
set -uo pipefail

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
PY="/data/vjuicefs_sz_ocr_wl/public_data/11193960/envs/jepa311/bin/python"
RUN="${EVAL_RUN:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/runs/exp10_curated/exp10-scale-20260717-140637-bbdc9c3}"
STEP="${EVAL_STEP:-500}"
MAXCLIPS="${EVAL_MAXCLIPS:-0}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ARMS=(sft_s0 sft_s1 mse_s0 mse_s1)
RES="$RUN/eval_step${STEP}"
mkdir -p "$RES"
cd "$PROJECT_ROOT"

cap=()
[[ "$MAXCLIPS" != "0" ]] && cap=(--max-clips "$MAXCLIPS" --seed 0)

one() {  # $1 arm  $2 task(MVBench|Tempcompass)  $3 data  $4 gpu
  local arm="$1" task="$2" data="$3" gpu="$4"
  local d="$RUN/outputs/exp10_curated_${arm}"
  local tag="${arm}_${task,,}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval \
    --config "$d/config.json" --ckpt "$d/step_${STEP}" \
    --data "$data" --task "$task" "${cap[@]}" \
    --output "$RES/${tag}.json" >"$RES/${tag}.log" 2>&1
  echo "[done] $tag rc=$? $(date '+%T')" >>"$RES/progress.log"
}

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"; [[ "$NGPU" -ge 1 ]] || NGPU=1
echo "== eval start $(date) step=$STEP maxclips=$MAXCLIPS ngpu=$NGPU ==" >"$RES/progress.log"
idx=0
for arm in "${ARMS[@]}"; do
  one "$arm" MVBench "$MVB" "$((idx % NGPU))" &
  idx=$((idx+1))
  one "$arm" Tempcompass "$TC" "$((idx % NGPU))" &
  idx=$((idx+1))
done
wait
echo "== all eval procs finished $(date) ==" >>"$RES/progress.log"

"$PY" - "$RES" "${ARMS[@]}" <<'PY' | tee "$RES/scorecard.txt"
import re, sys, os
res = sys.argv[1]; arms = sys.argv[2:]
def overall(logpath):
    if not os.path.exists(logpath): return "NA"
    val = "NA"
    for line in open(logpath, errors="ignore"):
        m = re.match(r"\s*overall:\s*([0-9.]+)", line)
        if m: val = m.group(1)
    return val
print(f"{'arm':<10} {'MVBench':>10} {'TempCompass':>12}")
for a in arms:
    print(f"{a:<10} {overall(os.path.join(res, a+'_mvbench.log')):>10} {overall(os.path.join(res, a+'_tempcompass.log')):>12}")
PY
touch "$RES/eval.done"
echo "=== EXP-10 step_${STEP} eval complete -> $RES/scorecard.txt ==="
