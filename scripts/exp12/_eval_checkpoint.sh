#!/usr/bin/env bash
set -Eeuo pipefail

STEP="${1:?usage: _eval_checkpoint.sh 400|800 [arm]}"
ONLY_ARM="${2:-}"
[[ "$STEP" == 400 || "$STEP" == 800 ]] || { echo "step must be 400 or 800" >&2; exit 2; }
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
RUN_ROOT="${EXP12_RUN_ROOT:?EXP12_RUN_ROOT is required}"
ROOT="${RUN_ROOT}/results/exp12_orca_token_sweep"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ARMS=(a0_ce_k4 a1_query_k4 a2_ce_k16 a3_query_k16 a4_ce_k64 a5_query_k64)
if [[ -n "$ONLY_ARM" ]]; then
  case "$ONLY_ARM" in
    a0_ce_k4|a1_query_k4|a2_ce_k16|a3_query_k16|a4_ce_k64|a5_query_k64) ;;
    *) echo "unknown EXP-12 arm: $ONLY_ARM" >&2; exit 2 ;;
  esac
  ARMS=("$ONLY_ARM")
fi
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
      tmp_subset="$lock/subset.jsonl"; tmp_ids="$lock/ids.tsv"
      "$PY" scripts/exp12/make_tempcompass_subset.py \
        --input "$TC" --output "$tmp_subset" --ids "$tmp_ids" \
        > "$ROOT/eval/subset_build.log"
      mv "$tmp_subset" "$subset"
      mv "$tmp_ids" "$ids"
      mv "$tmp_subset.sha256" "$subset.sha256"
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

run_arm_eval() {
  arm="$1"; gpu="$2"; out="$ROOT/$arm"
  [[ -f "$out/checkpoint-${STEP}/state.pt" ]] || { echo "missing $arm checkpoint-$STEP" >&2; return 1; }
  for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"; data="${DATASETS[$index]}"; suffix="${SUFFIXES[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m jepa_vlm.probes.mcq_eval \
      --config "$out/config.json" --ckpt "$out/checkpoint-${STEP}" \
      --data "$data" --task "$task" --output "$out/checkpoint-${STEP}_${suffix}.json" \
      > "$out/checkpoint-${STEP}_${suffix}.eval.log" 2>&1
  done
}

pids=(); failed=0; slot="${EXP12_EVAL_GPU:-0}"
for arm in "${ARMS[@]}"; do
  run_arm_eval "$arm" "$slot" & pids+=("$!"); slot=$((slot + 1))
  if [[ -z "$ONLY_ARM" && "${#pids[@]}" -eq 4 ]]; then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=(); slot=0
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo "checkpoint-$STEP evaluation failed" >&2; exit 1; }

for arm in "${ARMS[@]}"; do
  out="$ROOT/$arm"
  "$PY" - "$out/evaluator_config.json" "$STEP" "$MVB" "$TC" "$PROJECT_ROOT" <<'PY'
import hashlib, json, os, subprocess, sys
out, step, mvb, tc, project = sys.argv[1:]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
payload={"checkpoint":int(step), "scoring":"answer_likelihood_mean_token_ce",
         "evaluator":"jepa_vlm.probes.mcq_eval", "evaluator_commit":subprocess.check_output(
             ["git","-C",project,"rev-parse","HEAD"], text=True).strip(),
         "mvbench":{"path":mvb,"sha256":digest(mvb)},
         "tempcompass":{"path":tc,"sha256":digest(tc)}}
json.dump(payload, open(out,'w'), indent=2)
PY
done
echo "[exp12-eval] PASS checkpoint-$STEP"
