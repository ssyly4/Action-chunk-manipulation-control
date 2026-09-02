#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "[TOPPRA AB] compatibility entry; use run_30k_receding_toppra.sh" >&2
exec "$ROOT/run_30k_receding_toppra.sh" "$@"
