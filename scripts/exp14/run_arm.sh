#!/usr/bin/env bash
set -Eeuo pipefail

ARM="${1:?usage: run_arm.sh <b0_ce_seed1...b5_query_beatcopy_seed1>}"
case "$ARM" in
  b0_ce_seed1|b1_query_seed1|b2_noquery_seed0|b3_noquery_seed1|\
  b4_query_beatcopy_seed0|b5_query_beatcopy_seed1) ;;
  *) echo "unknown EXP-14 arm: $ARM" >&2; exit 2 ;;
esac
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
RUN_ROOT="${EXP14_RUN_ROOT:?EXP14_RUN_ROOT is required}"
MAX_STEPS="${EXP14_MAX_STEPS:-800}"
GRAD_ACCUM="${EXP14_GRAD_ACCUM:-4}"
NUM_WORKERS="${EXP14_NUM_WORKERS:-4}"
OUT="${RUN_ROOT}/results/exp14_state_diagnostics/${ARM}"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
mkdir -p "$OUT"
current_commit="${EXP14_GIT_COMMIT:-$(git rev-parse HEAD)}"

valid_checkpoint() {
  "$PY" - "$1" <<'PY'
import json, os, sys
path = sys.argv[1]
meta = os.path.join(os.path.dirname(path), "checkpoint_meta.json")
if os.path.isfile(meta):
    value = json.load(open(meta))
    ok = value.get("step_unit") == "optimizer_update" and value.get("state_bytes") == os.path.getsize(path)
    raise SystemExit(0 if ok else 1)
import torch
try: state = torch.load(path, map_location="cpu", weights_only=False)
except Exception: raise SystemExit(1)
raise SystemExit(0 if state.get("step_unit") == "optimizer_update" else 1)
PY
}

latest=""; latest_step=-1
for candidate in "$OUT"/checkpoint-*; do
  [[ -f "$candidate/state.pt" ]] || continue
  valid_checkpoint "$candidate/state.pt" || continue
  candidate_step="${candidate##*-}"
  [[ "$candidate_step" =~ ^[0-9]+$ ]] || continue
  if (( candidate_step > latest_step )); then
    latest="$candidate"; latest_step="$candidate_step"
  fi
done
if [[ -f "$OUT/checkpoint-${MAX_STEPS}/state.pt" ]] && valid_checkpoint "$OUT/checkpoint-${MAX_STEPS}/state.pt"; then
  if [[ -s "$OUT/git_commit.txt" && "$(<"$OUT/git_commit.txt")" != "$current_commit" ]]; then
    echo "$ARM completed under a different commit; use a new run ID" >&2
    exit 1
  fi
  echo "[exp14-arm] $ARM already complete"
  exit 0
fi
if [[ -n "$latest" && "${EXP14_RESUME:-0}" != 1 ]]; then
  echo "$ARM has an incomplete prior run; set EXP14_RESUME=1" >&2
  exit 1
fi
if [[ -z "$latest" && -s "$OUT/trainer_log.jsonl" ]]; then
  echo "$ARM has logs but no valid checkpoint; use a new run ID" >&2
  exit 1
fi
if [[ -n "$latest" ]]; then
  "$PY" - "$OUT/trainer_log.jsonl" "$latest_step" <<'PY'
import json, os, sys
path, checkpoint_step = sys.argv[1], int(sys.argv[2])
if not os.path.isfile(path): raise SystemExit(0)
kept = []
with open(path) as handle:
    for line in handle:
        if line.strip() and int(json.loads(line).get("step", 0)) <= checkpoint_step:
            kept.append(line)
tmp = path + ".resume-pruned.tmp"
with open(tmp, "w") as handle:
    handle.writelines(kept); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp, path)
PY
fi

overrides="train.output_dir=${OUT} train.max_steps=${MAX_STEPS} train.save_every=400 train.grad_accum=${GRAD_ACCUM} train.num_workers=${NUM_WORKERS}"
[[ -n "$latest" ]] && overrides+=" train.resume=${latest}"
printf '%s\n' "${EXP14_RUN_ID:-exp14}-${ARM}" > "$OUT/logical_job_id.txt"
[[ -f "$OUT/initial_git_commit.txt" ]] || printf '%s\n' "$current_commit" > "$OUT/initial_git_commit.txt"
printf '%s\n' "$current_commit" > "$OUT/git_commit.txt"
printf '%s\t%s\t%s\n' "$(date -Is)" "$current_commit" "${latest:-fresh}" >> "$OUT/attempt_history.tsv"

CONFIG="configs/exp14_state_diagnostics/${ARM}.yaml" GRAD_ACCUM="$GRAD_ACCUM" \
  EXTRA_OVERRIDES="$overrides" bash scripts/cluster/train_multinode.sh \
  2>&1 | tee -a "$OUT/launcher_rank${NODE_RANK:-0}.log"
[[ -f "$OUT/checkpoint-${MAX_STEPS}/state.pt" ]] || {
  nvidia-smi > "$OUT/failure_nvidia_smi.txt" 2>&1 || true
  echo "$ARM did not reach checkpoint-${MAX_STEPS}" >&2
  exit 1
}
if [[ "${NODE_RANK:-0}" == 0 ]]; then
  find "$OUT/tb" -maxdepth 1 -name 'events.out.tfevents.*' -exec ln -sf {} "$OUT/" \; 2>/dev/null || true
fi
