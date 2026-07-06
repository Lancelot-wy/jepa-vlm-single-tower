#!/usr/bin/env bash
set -euo pipefail
# (Re)build a single REAL-file folder that the platform TensorBoard (vtraining
# tensorboard -l <dir>) can serve as one merged view of the video JEPA runs.
#
# Uses HARD LINKS (same juicefs inode) so the platform TB reads real files AND
# sees live growth of still-training runs -- symlinks do NOT work (TB won't
# follow them) and plain copies would be static snapshots.
#
# Re-run after any run is restarted (a restart creates a new event file/inode).
#
# Usage: scripts/build_merged_tb.sh [run ...]   (default: the 5 key runs)

OUT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/outputs
MERGED="$OUT/tb_video_ablations"
RUNS=("${@:-v21 v1 mask75 mtp_off frozen_vit}")
# shellcheck disable=SC2206
RUNS=(${RUNS[*]})

rm -rf "$MERGED"; mkdir -p "$MERGED"
for r in "${RUNS[@]}"; do
  src="$OUT/jepa_llava_video_$r/tb"
  [[ -d "$src" ]] || { echo "!! no tb dir for $r ($src) -- skipping" >&2; continue; }
  mkdir -p "$MERGED/$r"
  # keep only the newest event file per run (avoids duplicate curves)
  newest="$(ls -t "$src"/events.out.tfevents.* 2>/dev/null | head -1 || true)"
  [[ -n "$newest" ]] || { echo "!! no events for $r" >&2; continue; }
  ln -f "$newest" "$MERGED/$r/$(basename "$newest")"
  echo "linked $r -> $(basename "$newest")"
done
echo "merged TB folder: $MERGED"
