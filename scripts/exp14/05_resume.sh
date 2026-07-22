#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ID="${1:?usage: $0 <existing-run-id>}"
exec bash "$(dirname "$0")/03_submit.sh" --resume --run-id "$RUN_ID"
