#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
RUN_ROOT="${EXP14_RUN_ROOT:-${BASE}/runs/exp14/smoke}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
ARMS=(b0_ce_seed1 b1_query_seed1 b2_noquery_seed0 b4_query_beatcopy_seed0)
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
[[ "$(nvidia-smi -L | grep -c '^GPU ' || true)" == 4 ]] || { echo "smoke needs 4 GPUs" >&2; exit 1; }

on_fail() {
  code=$?
  nvidia-smi > "$RUN_ROOT/smoke_failure_nvidia_smi.txt" 2>&1 || true
  echo "[exp14-smoke] failed; formal batch and loss weights were not changed" >&2
  exit "$code"
}
trap on_fail ERR

port=29740
for arm in "${ARMS[@]}"; do
  out="$RUN_ROOT/smoke/$arm"
  rm -rf "$out"; mkdir -p "$out"
  common="train.output_dir=${out} train.num_workers=0 train.grad_accum=8 train.log_every=1 train.eval_every=999999"
  CONFIG="configs/exp14_state_diagnostics/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" GRAD_ACCUM=8 \
    EXTRA_OVERRIDES="$common train.max_steps=1 train.save_every=1" \
    bash scripts/cluster/train_multinode.sh > "$out/first_step_launcher.log" 2>&1
  CONFIG="configs/exp14_state_diagnostics/${arm}.yaml" NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" GRAD_ACCUM=8 \
    EXTRA_OVERRIDES="$common train.max_steps=2 train.save_every=1 train.resume=${out}/checkpoint-1" \
    bash scripts/cluster/train_multinode.sh > "$out/resume_launcher.log" 2>&1
  [[ -f "$out/checkpoint-2/state.pt" ]] || { echo "$arm did not resume to step 2" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$PY" -m jepa_vlm.probes.mcq_eval \
    --config "$out/config.json" --ckpt "$out/checkpoint-2" --data "$TC" \
    --task Tempcompass --max-clips 2 --output "$out/tempcompass_smoke.json" \
    > "$out/tempcompass_eval.log" 2>&1
  "$PY" - "$out" "$arm" <<'PY'
import json, math, os, sys, torch
root, arm = sys.argv[1:]
cfg = json.load(open(os.path.join(root, "config.json")))
logs = [json.loads(line) for line in open(os.path.join(root, "trainer_log.jsonl")) if line.strip()]
if not logs or any(not math.isfinite(float(row["loss"])) for row in logs):
    raise SystemExit("missing/non-finite loss")
last = logs[-1]
if int(last["model/visual_tokens_per_unit"]) != 64:
    raise SystemExit("K64 contract failed")
mode = cfg["model"]["state_predictor_mode"]
if mode != "none":
    for key in ("state/centered_margin", "state/persistence_ratio", "state/beat_copy_loss"):
        if key not in last or not math.isfinite(float(last[key])):
            raise SystemExit(f"missing state metric {key}")
    expected_query = 0.0 if mode == "no_query" else 1.0
    if float(last["state/query_enabled"]) != expected_query:
        raise SystemExit("query/no-query path mismatch")
if "beatcopy" in arm and cfg["model"]["beat_copy_loss_weight"] != 1.0:
    raise SystemExit("anti-copy weight mismatch")
one = torch.load(os.path.join(root, "checkpoint-1/state.pt"), map_location="cpu", weights_only=False)
two = torch.load(os.path.join(root, "checkpoint-2/state.pt"), map_location="cpu", weights_only=False)
if mode != "none" and int(two["model_aux"]["state_center"]["updates"]) <= int(one["model_aux"]["state_center"]["updates"]):
    raise SystemExit("running center did not restore/update")
json.dump({"arm": arm, "status": "PASS"}, open(os.path.join(root, "smoke_status.json"), "w"), indent=2)
PY
  port=$((port + 1))
done
trap - ERR
echo "[exp14-smoke] PASS CE/query/no-query/anti-copy, save/resume/eval"
