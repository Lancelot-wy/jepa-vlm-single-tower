#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
MANIFEST="${EXP14_MANIFEST:-${BASE}/jepa_data/exp10_curated/qa_train_clean.jsonl}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
RUN_ROOT="${EXP14_RUN_ROOT:-${BASE}/runs/exp14/preflight-$(date +%Y%m%d-%H%M%S)}"
EXPECTED_COMMIT="${EXP14_GIT_COMMIT:-}"

die() { echo "[exp14-preflight] ERROR: $*" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/.git" ]] || die "missing repository: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -z "$(git status --porcelain)" ]] || die "checkout must be clean and committed"
HEAD="$(git rev-parse HEAD)"
[[ -z "$EXPECTED_COMMIT" || "$HEAD" == "$EXPECTED_COMMIT" ]] || die "HEAD $HEAD != $EXPECTED_COMMIT"
[[ -d "$MODEL_ROOT" && -s "$MANIFEST" && -s "$MVB" && -s "$TC" ]] || die "model/data path missing"
mkdir -p "$RUN_ROOT/preflight"
# shellcheck disable=SC1091
source scripts/cluster/env.cluster.sh
PY="${JEPA_ENV}/bin/python"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
[[ "$GPU_COUNT" -eq 4 ]] || die "each Worker must expose 4 GPUs; found $GPU_COUNT"

args=(
  --project "$PROJECT_ROOT" --manifest "$MANIFEST" --model "$MODEL_ROOT"
  --mvbench "$MVB" --tempcompass "$TC" --output "$RUN_ROOT/preflight"
  --world-size 8 --grad-accum 4
  --config-dir configs/exp14_state_diagnostics
  --arms b0_ce_seed1 b1_query_seed1 b2_noquery_seed0 b3_noquery_seed1 b4_query_beatcopy_seed0 b5_query_beatcopy_seed1
  --expected-k 64 64 64 64 64 64
  --expected-modes none query no_query no_query query query
)
[[ -n "$EXPECTED_COMMIT" ]] && args+=(--expected-commit "$EXPECTED_COMMIT")
[[ "${EXP14_PREFLIGHT_LOAD_MODEL:-1}" == 1 ]] && args+=(--load-model)
"$PY" scripts/exp12/runtime_preflight.py "${args[@]}" | tee "$RUN_ROOT/preflight/runtime.log"
echo "$HEAD" > "$RUN_ROOT/preflight/git_commit.txt"
sha256sum "$MANIFEST" | awk '{print $1}' > "$RUN_ROOT/preflight/manifest.sha256"
echo "[exp14-preflight] PASS commit=$HEAD world=8 grad_accum=4 effective_batch=32"
