#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# B mode preserves the policy path. The current follower command remains the
# handoff boundary but is not inserted as an extra waypoint into q_ref(s).
export NERO_TOPPRA_BOUNDARY_PATH_MODE=path_curvature
export NERO_TOPPRA_CURVATURE_SPEED_MARGIN="${NERO_TOPPRA_CURVATURE_SPEED_MARGIN:-0.80}"

exec "$ROOT/run_30k_receding_toppra.sh" "$@"
