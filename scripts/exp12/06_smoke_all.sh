#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
bash "$root/03_smoke_k4.sh"
bash "$root/04_smoke_k16.sh"
bash "$root/05_smoke_k64.sh"
echo "[exp12-smoke] all K values passed"
