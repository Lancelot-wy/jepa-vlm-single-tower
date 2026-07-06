#!/usr/bin/env bash
set -uo pipefail

# Temporal linear-probe matrix (EXPERIMENTS.md section 2) as a single-GPU platform job.
# Runs on rank0/one GPU: for each model (base + 5 round-2 arms) and each temporal
# transform (random_shuffle / random_reverse) it extracts train+val features and
# fits a linear probe on layer27_frames, appending "model transform: acc" to
# $FEATS/probe_results.txt. Diving48 class probe is intentionally skipped (source
# unreachable); add it here once the data lands on the shared disk.
#
# Local runs hit the tool's per-command timeout and can't background, so probing
# is done here on the cluster instead. Submit with a 1-node/1-GPU job whose command
# is:  bash scripts/cluster/run_probes.sh
#
# Overridable env: MAXTR (train clips, default 4000), FEATURE (default layer27_frames).

PROJECT_ROOT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data"
OUT="${OUTPUT_ROOT}"
FEATS="${OUT}/feats"
mkdir -p "$FEATS"
TRAIN="$DATA/llava_video/train.jsonl"
VAL="$DATA/llava_video/val.jsonl"
MAXTR="${MAXTR:-4000}"
FEATURE="${FEATURE:-layer27_frames}"
RESULTS="$FEATS/probe_results.txt"
echo "# temporal probe results  $(date)" >> "$RESULTS"

# name|config|ckpt   (empty ckpt = untrained base model)
MODELS=(
  "base|configs/r2_v21.yaml|"
  "r2_frozen|$OUT/r2_frozen/config.json|$OUT/r2_frozen/step_2000"
  "r2_v21|$OUT/r2_v21/config.json|$OUT/r2_v21/step_2000"
  "r2_varreg|$OUT/r2_varreg/config.json|$OUT/r2_varreg/step_2000"
  "r2_residual|$OUT/r2_residual/config.json|$OUT/r2_residual/step_2000"
  "r2_sft_baseline|$OUT/r2_sft_baseline/config.json|$OUT/r2_sft_baseline/step_2000"
)

extract() {  # $1=config $2=ckptflag $3=manifest $4=transform $5=out [maxclips]
  if [[ -f "$5" ]]; then echo "   $5 exists -> skip"; return 0; fi
  local cap=()
  [[ -n "${6:-}" ]] && cap=(--max-clips "$6")
  python -m jepa_vlm.probes.extract_features \
    --config "$1" $2 --manifest "$3" --temporal-transform "$4" \
    --out "$5" "${cap[@]}"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name config ckpt <<< "$entry"
  # ONLY_MODEL lets one job handle one model so the matrix runs in parallel.
  if [[ -n "${ONLY_MODEL:-}" && "$ONLY_MODEL" != "$name" ]]; then continue; fi
  if [[ -n "$config" && "$config" == *.json && ! -f "$config" ]]; then
    echo "!! $name: missing $config -- skipping (train not finished?)" | tee -a "$RESULTS"; continue
  fi
  ckflag=""
  [[ -n "$ckpt" ]] && ckflag="--ckpt $ckpt"
  for T in random_shuffle random_reverse; do
    tr="$FEATS/${name}_${T}_tr.pt"; va="$FEATS/${name}_${T}_va.pt"
    echo ">> [$name/$T] extracting features"
    extract "$config" "$ckflag" "$TRAIN" "$T" "$tr" "$MAXTR" || { echo "!! $name $T: extract-train FAILED" | tee -a "$RESULTS"; continue; }
    extract "$config" "$ckflag" "$VAL"   "$T" "$va" ""       || { echo "!! $name $T: extract-val FAILED"   | tee -a "$RESULTS"; continue; }
    acc="$(python -m jepa_vlm.probes.linear_probe --train "$tr" --val "$va" --feature "$FEATURE" 2>&1 | tail -1)"
    echo "$name $T: $acc" | tee -a "$RESULTS"
  done
done

echo "=== probe matrix done -> $RESULTS ==="
cat "$RESULTS"
