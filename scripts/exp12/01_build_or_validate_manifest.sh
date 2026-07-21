#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-/data/vjuicefs_sz_ocr_wl/public_data/11193960}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/jepa-vlm-single-tower}"
MANIFEST="${EXP12_MANIFEST:-${BASE}/jepa_data/exp10_curated/qa_train_clean.jsonl}"
RUN_ROOT="${EXP12_RUN_ROOT:-${BASE}/runs/exp12/manifest-audit}"
cd "$PROJECT_ROOT"

if [[ ! -s "$MANIFEST" ]]; then
  echo "[exp12-manifest] frozen EXP-10 manifest absent; invoking audited builder"
  PROJECT="$PROJECT_ROOT" DATA_ROOT="$(dirname "$MANIFEST")" \
    bash scripts/direct/run_exp10_curated_4gpu.sh prep
fi
[[ -s "$MANIFEST" ]] || { echo "manifest build failed: $MANIFEST" >&2; exit 1; }
mkdir -p "$RUN_ROOT/manifest"
digest="$(sha256sum "$MANIFEST" | awk '{print $1}')"
if [[ -f "$RUN_ROOT/manifest/manifest.sha256" ]]; then
  old="$(<"$RUN_ROOT/manifest/manifest.sha256")"
  [[ "$old" == "$digest" ]] || { echo "frozen manifest changed: $old -> $digest" >&2; exit 1; }
fi
printf '%s\n' "$digest" > "$RUN_ROOT/manifest/manifest.sha256"
python_bin="${JEPA_ENV:+${JEPA_ENV}/bin/python}"; python_bin="${python_bin:-python3}"
"$python_bin" - "$MANIFEST" "$RUN_ROOT/manifest/summary.json" <<'PY'
import collections, json, os, sys
path, out = sys.argv[1:]
sources = collections.Counter(); rows = 0; missing = 0; malformed = 0; duplicate_ids = 0
seen_ids = set()
with open(path) as handle:
    for line in handle:
        if not line.strip(): continue
        row = json.loads(line); rows += 1
        sources[row.get("source_dataset", "<missing>")] += 1
        missing += int(not os.path.isfile(row.get("video", "")))
        malformed += int(any(not row.get(key) for key in ("video", "question", "answer", "source_id")))
        identity = (
            row.get("source_dataset"), row.get("source_id"),
            row.get("video"), row.get("question"),
        )
        duplicate_ids += int(identity in seen_ids)
        seen_ids.add(identity)
result = {"manifest": os.path.abspath(path), "rows": rows,
          "sources": dict(sources), "missing_video_paths": missing,
          "malformed_rows": malformed, "duplicate_source_ids": duplicate_ids}
if rows == 0 or "<missing>" in sources or missing or malformed or duplicate_ids:
    raise SystemExit(f"manifest validation failed: {result}")
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
PY
echo "[exp12-manifest] PASS sha256=$digest"
