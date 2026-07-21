#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname "$0")/_smoke_one.sh" 16
