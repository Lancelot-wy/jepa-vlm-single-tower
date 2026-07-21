#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname "$0")/_eval_checkpoint.sh" 400
