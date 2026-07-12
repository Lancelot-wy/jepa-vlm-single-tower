#!/usr/bin/env bash
set -uo pipefail

# Concatenate + extract the OVO-Bench split tars into OVO_ROOT so that
# streaming_eval.py (bench=ovo) finds video-root/chunked_videos/<id>.mp4.
# The HF release ships chunked_videos.tar.part{aa..ao} (~143GB) and
# src_videos.tar.part{aa..ae} (~44GB); parts glob-sort correctly. Only
# chunked_videos is needed for EPM/ASI eval; SRC_VIDEOS=1 also extracts src.
# Run as a CPU platform job.

SRC="${SRC:-/data/vjuicefs_ai_gpt_vision_wl02/public_data/open_source_data/auto_download/huggingface/11193960/JoeLeelyf_OVO-Bench/OVO-Bench}"
DEST="${DEST:-/data/vjuicefs_sz_ocr_wl/public_data/11193960/stream/ovo}"
mkdir -p "$DEST"

echo ">> extracting chunked_videos.tar.part* -> $DEST"
cat "$SRC"/chunked_videos.tar.part* | tar xf - -C "$DEST" || echo "!! chunked_videos extract failed"

if [[ "${SRC_VIDEOS:-0}" == "1" ]]; then
  echo ">> extracting src_videos.tar.part* -> $DEST"
  cat "$SRC"/src_videos.tar.part* | tar xf - -C "$DEST" || echo "!! src_videos extract failed"
fi

echo "=== top-level entries under $DEST ==="
ls -1 "$DEST"
echo "=== chunked_videos mp4 count ==="
find "$DEST" -path '*chunked_videos*' -name '*.mp4' | wc -l
echo "=== OVO extraction done -> $DEST ==="
