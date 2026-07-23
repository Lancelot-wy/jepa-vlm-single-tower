#!/usr/bin/env bash
# Initial server-side gate for the EXP-15 coding Agent. This does not submit GPUs.
set -Eeuo pipefail

ROOT_DEFAULT="/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DEFAULT}"
EXPECTED_BRANCH="${EXP15_BRANCH:-exp15-native-orca}"
BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
STAGE="implementation"
SKIP_SERVER_PATHS=0

die() { echo "[exp15-agent-preflight] ERROR: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) shift; [[ $# -gt 0 ]] || die "--stage requires implementation or launch"; STAGE="$1" ;;
    --skip-server-paths) SKIP_SERVER_PATHS=1 ;;
    -h|--help) sed -n '1,38p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
case "$STAGE" in implementation|launch) ;; *) die "bad stage: $STAGE" ;; esac

[[ -d "$PROJECT_ROOT/.git" ]] || die "repository missing: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
BRANCH="$(git branch --show-current)"
HEAD="$(git rev-parse HEAD)"
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || die "expected branch $EXPECTED_BRANCH, got $BRANCH"
[[ -z "$(git status --porcelain)" ]] || die "starting checkout is dirty; preserve and resolve changes first"

PY="${JEPA_ENV:+${JEPA_ENV}/bin/python}"
[[ -n "$PY" && -x "$PY" ]] || PY="${BASE}/envs/jepa311/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -x "$PY" ]] || die "Python not found"

STAMP="$(date '+%Y%m%d-%H%M%S')"
REPORT_ROOT="${EXP15_PREFLIGHT_ROOT:-${BASE}/runs/exp15/agent-preflight-${STAMP}-${HEAD:0:8}}"
mkdir -p "$REPORT_ROOT"

CONTRACT_STAGE="handoff"
[[ "$STAGE" == "launch" ]] && CONTRACT_STAGE="launch"
"$PY" scripts/exp15/validate_contract.py "$CONTRACT_STAGE" \
  --json-output "$REPORT_ROOT/handoff_contract.json"

{
  echo "time=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "host=$(hostname)"
  echo "project_root=$PROJECT_ROOT"
  echo "branch=$BRANCH"
  echo "commit=$HEAD"
  echo "stage=$STAGE"
  echo "python=$PY"
  echo "vtraining=/data/vtraining_04/code/vtraining/cli/vtraining"
  git status --short
  git log -12 --oneline
  uname -a
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi -L
  else
    echo "nvidia-smi=unavailable"
  fi
} | tee "$REPORT_ROOT/environment.txt"

"$PY" - <<'PY' | tee "$REPORT_ROOT/python_packages.txt"
import importlib
import os
missing = []
for name in ("torch", "transformers", "accelerate", "av", "yaml"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "version-unknown"))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))
        missing.append(name)
if missing and os.environ.get("EXP15_ALLOW_MISSING_DEPS") != "1":
    raise SystemExit("missing required Python packages: " + ", ".join(missing))
PY

if [[ "$SKIP_SERVER_PATHS" == 0 ]]; then
  "$PY" - "$PROJECT_ROOT/contracts/exp15.yaml" "$REPORT_ROOT/path_audit.json" <<'PY'
from __future__ import annotations
import glob
import json
import os
from pathlib import Path
import sys
import yaml

contract_path, output_path = map(Path, sys.argv[1:])
contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
paths = contract["server_paths"]
checks = {
    "model_root": (paths["model_root"], "dir"),
    "llava_video_root": (paths["llava_video_root"], "dir"),
    "vript_metadata_glob": (paths["vript_metadata_glob"], "glob"),
    "internvid_metadata": (paths["internvid_metadata"], "file"),
    "mvbench": (paths["mvbench"], "file"),
    "tempcompass": (paths["tempcompass"], "file"),
}
report = {}
failed = []
for name, (value, kind) in checks.items():
    if kind == "glob":
        matches = sorted(glob.iglob(value))
        ok = bool(matches)
        detail = {"pattern": value, "matches": len(matches), "first": matches[:3]}
    else:
        path = Path(value)
        ok = path.is_dir() if kind == "dir" else path.is_file() and path.stat().st_size > 0
        detail = {"path": value, "kind": kind, "readable": os.access(path, os.R_OK)}
        if path.exists():
            detail["size"] = path.stat().st_size
    detail["ok"] = ok
    report[name] = detail
    if not ok:
        failed.append(name)

run_root = Path(paths["run_root"])
run_root.mkdir(parents=True, exist_ok=True)
probe = run_root / ".exp15-write-probe"
probe.write_text("ok\n", encoding="utf-8")
probe.unlink()
report["run_root"] = {"path": str(run_root), "ok": True, "writable": True}
output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
if failed:
    raise SystemExit("missing required server paths: " + ", ".join(failed))
PY
fi

VTRAINING="/data/vtraining_04/code/vtraining/cli/vtraining"
if [[ "$STAGE" == "launch" ]]; then
  [[ -x "$VTRAINING" ]] || die "launch CLI missing: $VTRAINING"
elif [[ ! -x "$VTRAINING" ]]; then
  echo "[exp15-agent-preflight] WARN: vtraining CLI not present on this development host" | tee -a "$REPORT_ROOT/environment.txt"
fi

echo "$HEAD" > "$REPORT_ROOT/git_commit.txt"
echo "[exp15-agent-preflight] PASS report=$REPORT_ROOT"
echo "[exp15-agent-preflight] next: read results/exp15/AGENT_STATUS.md and start S1"
