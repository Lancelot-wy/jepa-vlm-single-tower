#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ID="${1:?usage: $0 <existing-run-id>}"
exec bash "$(dirname "$0")/07_submit_a0_a5.sh" --resume --run-id "$RUN_ID"
