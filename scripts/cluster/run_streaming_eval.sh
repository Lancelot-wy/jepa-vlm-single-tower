#!/usr/bin/env bash
set -uo pipefail

# V4 streaming-eval worker (single 4xL40S node; run inside a platform pod).
# Evaluates a list of arms (base + finished ckpts) on OVO backward + StreamingBench,
# 4 arms in parallel (one per GPU). Resume-safe: per-question jsonl skips done qids.
#
#   OVO_ROOT=<dir containing ovo_bench_new.json + chunked_videos/>       (required)
#   SB_CSV=<comma-separated StreamingBench csv paths>                    (required)
#   SB_VIDEO_ROOT=<dir containing sample_N/video.mp4 for those csvs>     (required)
#   ARMS="v4_ctrl_s0 v4_dv25_s0 ..."   STEP=step_4000   MODES="recent prefix"
#
#   source scripts/cluster/env.cluster.sh
#   OVO_ROOT=... SB_CSV=... SB_VIDEO_ROOT=... ARMS="base v4_ctrl_s0" \
#     bash scripts/cluster/run_streaming_eval.sh

BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
PROJECT="$BASE/jepa-vlm-single-tower"
OUTROOT="$BASE/outputs"
RESULTS="$BASE/outputs/streaming_eval"
mkdir -p "$RESULTS"
cd "$PROJECT"

: "${OVO_ROOT:?set OVO_ROOT}"; : "${SB_CSV:?set SB_CSV}"; : "${SB_VIDEO_ROOT:?set SB_VIDEO_ROOT}"
ARMS="${ARMS:?set ARMS (space-separated run names; 'base' = untrained)}"
STEP="${STEP:-step_4000}"
MODES="${MODES:-recent prefix}"
REF_CONFIG="${REF_CONFIG:-configs/v4_ctrl_s0.yaml}"   # config for arm 'base'

gpu=0
for arm in $ARMS; do
  if [[ "$arm" == "base" ]]; then
    cfg="$REF_CONFIG"; ckpt=""
  else
    cfg="$OUTROOT/$arm/config.json"; ckpt="$OUTROOT/$arm/$STEP"
    if [[ ! -d "$ckpt" ]]; then echo "!! $ckpt missing -- skip $arm"; continue; fi
  fi
  for mode in $MODES; do
    (
      export CUDA_VISIBLE_DEVICES=$gpu
      echo ">> [gpu$gpu] $arm ovo/$mode"
      python -m jepa_vlm.probes.streaming_eval --config "$cfg" --ckpt "$ckpt" \
        --bench ovo --data "$OVO_ROOT/ovo_bench_new.json" --video-root "$OVO_ROOT" \
        --tasks EPM,ASI --mode "$mode" --window "${WINDOW:-64}" \
        --out "$RESULTS/ovo_${arm}_${mode}.jsonl" \
        > "$RESULTS/ovo_${arm}_${mode}.log" 2>&1
      echo ">> [gpu$gpu] $arm sb/$mode"
      python -m jepa_vlm.probes.streaming_eval --config "$cfg" --ckpt "$ckpt" \
        --bench sb --data "$SB_CSV" --video-root "$SB_VIDEO_ROOT" \
        --mode "$mode" --window "${WINDOW:-64}" --max-items "${SB_MAX:-500}" \
        --out "$RESULTS/sb_${arm}_${mode}.jsonl" \
        > "$RESULTS/sb_${arm}_${mode}.log" 2>&1
    ) &
    gpu=$(( (gpu + 1) % 4 ))
    # at most 4 concurrent workers (one per GPU)
    while [[ $(jobs -rp | wc -l) -ge 4 ]]; do wait -n; done
  done
done
wait
echo "ALL STREAMING EVALS DONE"
python scripts/summarize_streaming.py "$RESULTS" || true
