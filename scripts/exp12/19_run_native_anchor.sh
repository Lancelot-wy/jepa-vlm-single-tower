#!/usr/bin/env bash
set -Eeuo pipefail

# One-node / four-GPU resumable entrypoint.  Set MAX_CLIPS=20 and a separate
# EXP12_NATIVE_ANCHOR_ROOT for smoke; MAX_CLIPS=0 means the complete benchmarks.
bash scripts/exp12/15_native_anchor_preflight.sh
bash scripts/exp12/16_eval_custom_anchor.sh
bash scripts/exp12/17_eval_native_anchor.sh
bash scripts/exp12/18_collect_native_anchor.sh
