#!/usr/bin/env bash
set -uo pipefail

# Extract the StreamingBench task zips into per-task video roots that match
# streaming_eval.py's SB mode (video-root/sample_<N>/video.mp4). Each task's
# split zips are globally numbered (e.g. RTVU 1-50 .. 451-500 -> sample_1..500),
# so all splits of one task extract into a single dir. Strips __MACOSX junk.
# Run as a CPU platform job (see submit at bottom of the chat / inline yaml).

SRC="${SRC:-/data/vjuicefs_ai_gpt_vision_wl02/public_data/open_source_data/auto_download/huggingface/11193960/mjuicem_StreamingBench/StreamingBench}"
DEST="${DEST:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/stream/streamingbench}"
mkdir -p "$DEST"

shopt -s nullglob
for z in "$SRC"/*.zip; do
  base="$(basename "$z" .zip)"
  base="$(echo "$base" | sed -E 's/_[0-9]+-[0-9]+$//')"           # drop trailing "_1-50"
  grp="$(echo "$base" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+|_+$//g')"
  echo ">> $(basename "$z")  ->  $DEST/$grp"
  unzip -oq "$z" -d "$DEST/$grp" || echo "!! unzip failed: $z"
done

echo ">> cleaning mac junk"
find "$DEST" -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$DEST" -name '.DS_Store' -delete 2>/dev/null
find "$DEST" -name '._*' -delete 2>/dev/null

echo "=== per-task video counts ==="
for d in "$DEST"/*/; do
  printf '  %-40s %s videos\n' "$(basename "$d")" "$(find "$d" -name video.mp4 | wc -l)"
done
echo "=== StreamingBench extraction done -> $DEST ==="
