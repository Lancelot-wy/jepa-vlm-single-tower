#!/usr/bin/env bash
set -Eeuo pipefail

K="${1:?usage: _smoke_one.sh 4|16|64}"
case "$K" in
  4) ARMS=(a0_ce_k4 a1_query_k4); PORT=29604 ;;
  16) ARMS=(a2_ce_k16 a3_query_k16); PORT=29616 ;;
  64) ARMS=(a4_ce_k64 a5_query_k64); PORT=29664 ;;
  *) echo "K must be 4, 16, or 64" >&2; exit 2 ;;
esac
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
RUN_ROOT="${EXP12_RUN_ROOT:-${BASE}/runs/exp12/smoke}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
[[ "$(nvidia-smi -L | grep -c '^GPU ' || true)" == 4 ]] || { echo "smoke needs 4 GPUs" >&2; exit 1; }

on_fail() {
  code=$?
  nvidia-smi > "$RUN_ROOT/smoke_k${K}_failure_nvidia_smi.txt" 2>&1 || true
  echo "[exp12-smoke] K=$K failed; batch/cutoff were not changed" >&2
  exit "$code"
}
trap on_fail ERR

for arm in "${ARMS[@]}"; do
  out="$RUN_ROOT/smoke/k${K}/${arm}"
  rm -rf "$out"
  mkdir -p "$out"
  common="train.output_dir=${out} train.num_workers=0 train.grad_accum=8 train.log_every=1 train.eval_every=999999"
  CONFIG="configs/orca_token_sweep/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" GRAD_ACCUM=8 \
    EXTRA_OVERRIDES="$common train.max_steps=1 train.save_every=1" \
    bash scripts/cluster/train_multinode.sh 2>&1 | tee "$out/first_step_launcher.log"
  [[ -f "$out/checkpoint-1/state.pt" ]] || { echo "$arm did not save checkpoint-1" >&2; exit 1; }
  CONFIG="configs/orca_token_sweep/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" GRAD_ACCUM=8 \
    EXTRA_OVERRIDES="$common train.max_steps=2 train.save_every=1 train.resume=${out}/checkpoint-1" \
    bash scripts/cluster/train_multinode.sh 2>&1 | tee "$out/resume_launcher.log"
  [[ -f "$out/checkpoint-2/state.pt" ]] || { echo "$arm did not resume to checkpoint-2" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$PY" -m jepa_vlm.probes.mcq_eval \
    --config "$out/config.json" --ckpt "$out/checkpoint-2" --data "$TC" \
    --task Tempcompass --max-clips 2 --output "$out/tempcompass_smoke.json" \
    > "$out/tempcompass_eval.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "$PY" -m jepa_vlm.probes.mcq_eval \
    --config "$out/config.json" --ckpt "$out/checkpoint-2" --data "$MVB" \
    --task MVBench --max-clips 2 --output "$out/mvbench_smoke.json" \
    > "$out/mvbench_eval.log" 2>&1
  "$PY" - "$out" "$arm" "$K" <<'PY'
import json, math, os, sys, torch
root, arm, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
audit = json.load(open(os.path.join(root, "parameter_audit.json")))
if audit["visual_parameters_in_optimizer"] or audit["frozen_parameters_in_optimizer"]:
    raise SystemExit("optimizer audit failed")
logs = [json.loads(line) for line in open(os.path.join(root, "trainer_log.jsonl")) if line.strip()]
if not logs or any(not math.isfinite(float(row["loss"])) for row in logs):
    raise SystemExit("missing or non-finite training loss")
last = logs[-1]
for key in ("model/visual_tokens_per_unit", "model/deepstack_token_count",
            "model/pooled_grid_h", "model/pooled_grid_w"):
    if key not in last: raise SystemExit(f"missing shape metric {key}")
if int(last["model/visual_tokens_per_unit"]) != k:
    raise SystemExit("K metric mismatch")
if "query" in arm:
    required = ("state/centered_margin", "state/persistence_ratio", "state/target_std",
                "state/pred_std", "state/retrieval_top1", "state/retrieval_top5")
    for key in required:
        if key not in last or not math.isfinite(float(last[key])):
            raise SystemExit(f"missing/non-finite state metric {key}")
    one = torch.load(os.path.join(root, "checkpoint-1/state.pt"), map_location="cpu", weights_only=False)
    two = torch.load(os.path.join(root, "checkpoint-2/state.pt"), map_location="cpu", weights_only=False)
    u1 = int(one["model_aux"]["state_center"]["updates"])
    u2 = int(two["model_aux"]["state_center"]["updates"])
    if u2 <= u1: raise SystemExit("running center did not restore/update after resume")
json.dump({"arm": arm, "K": k, "steps": 2, "save": True, "resume": True,
           "eval": True, "status": "PASS"}, open(os.path.join(root, "smoke_status.json"), "w"), indent=2)
PY
done
trap - ERR
echo "[exp12-smoke] PASS K=$K CE+Query, 4-GPU DDP, save/resume/eval"
