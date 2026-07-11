#!/usr/bin/env bash
set -euo pipefail
# One-liner for the cluster side: commit lightweight result files back to GitHub
# on the `results-cluster` branch (summaries + per-question jsonl; NO checkpoints).
#   bash scripts/push_results.sh "a1+b3 done"
cd "$(dirname "$0")/.."
MSG="${1:-cluster results $(date +%F_%H%M)}"
git fetch origin
git checkout -B results-cluster origin/main 2>/dev/null || git checkout -b results-cluster
find results -maxdepth 3 -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.log" \) -size -50M \
  -exec git add -f {} + 2>/dev/null || true
git commit -m "results: ${MSG}" || { echo "nothing new to commit"; exit 0; }
git push -u origin results-cluster
echo "pushed to branch results-cluster"
