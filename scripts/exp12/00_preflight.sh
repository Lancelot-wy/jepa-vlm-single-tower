#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
MODEL_ROOT="${MODEL_ROOT:-${BASE}/models/Qwen3-VL-2B-Instruct}"
MANIFEST="${EXP12_MANIFEST:-${BASE}/jepa_data/exp10_curated/qa_train_clean.jsonl}"
MVB="${MVB:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl}"
TC="${TC:-/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl}"
RUN_ROOT="${EXP12_RUN_ROOT:-${BASE}/runs/exp12/preflight-$(date +%Y%m%d-%H%M%S)}"
EXPECTED_COMMIT="${EXP12_GIT_COMMIT:-}"

die() { echo "[exp12-preflight] ERROR: $*" >&2; exit 1; }
[[ -d "$PROJECT_ROOT/.git" ]] || die "missing repository: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
# `python scripts/exp12/x.py` only puts scripts/exp12/ on sys.path, not the repo
# root, and the cluster python does not add the CWD. Export the repo root so every
# python invocation here can `import jepa_vlm`.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -z "$(git status --porcelain)" ]] || die "checkout must be clean and committed"
HEAD="$(git rev-parse HEAD)"
[[ -z "$EXPECTED_COMMIT" || "$HEAD" == "$EXPECTED_COMMIT" ]] || die "HEAD $HEAD != fixed commit $EXPECTED_COMMIT"
[[ -d "$MODEL_ROOT" ]] || die "missing model path: $MODEL_ROOT"
[[ -s "$MANIFEST" ]] || die "missing manifest: $MANIFEST"
[[ -s "$MVB" && -s "$TC" ]] || die "missing evaluator data"
mkdir -p "$RUN_ROOT/preflight"

if [[ -f scripts/cluster/env.cluster.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/cluster/env.cluster.sh
fi
PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"
PY="${PY:-$(command -v python3 || true)}"
[[ -x "$PY" ]] || die "Python unavailable"
# Video decode uses the python `av`/`decord` bindings, not the ffmpeg CLI; the
# training image ships `av` but not the ffmpeg binary. Warn instead of dying on a
# missing CLI, and keep the hard dependency check on the `av` python module below.
if ! command -v ffmpeg >/dev/null; then
  echo "[exp12-preflight] WARN: ffmpeg CLI not on PATH (not used; decode via python av)"
fi
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
[[ "$GPU_COUNT" -eq 4 ]] || die "preflight Worker must expose exactly 4 GPUs; found $GPU_COUNT"

"$PY" - <<'PY' | tee "$RUN_ROOT/preflight/dependencies.txt"
import importlib.util
import torch, transformers, accelerate, av
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("av", av.__version__)
for name in ("torchvision", "decord", "llamafactory"):
    print(name, "available" if importlib.util.find_spec(name) else "not-installed (not used by this repo)")
PY

args=(
  --project "$PROJECT_ROOT" --manifest "$MANIFEST" --model "$MODEL_ROOT"
  --mvbench "$MVB" --tempcompass "$TC" --output "$RUN_ROOT/preflight"
  --world-size "$(( ${NPROC_PER_NODE:-$GPU_COUNT} * ${NNODES:-1} ))"
  --grad-accum "${EXP12_GRAD_ACCUM:-8}"
)
[[ -n "$EXPECTED_COMMIT" ]] && args+=(--expected-commit "$EXPECTED_COMMIT")
[[ "${EXP12_PREFLIGHT_LOAD_MODEL:-1}" == 1 ]] && args+=(--load-model)
"$PY" scripts/exp12/runtime_preflight.py "${args[@]}" | tee "$RUN_ROOT/preflight/runtime.log"

test_file="$(mktemp "$RUN_ROOT/.write-test.XXXXXX")"
rm -f "$test_file"
echo "$HEAD" > "$RUN_ROOT/preflight/git_commit.txt"
sha256sum "$MANIFEST" | awk '{print $1}' > "$RUN_ROOT/preflight/manifest.sha256"
echo "[exp12-preflight] PASS commit=$HEAD gpu=$GPU_COUNT output=$RUN_ROOT/preflight"
