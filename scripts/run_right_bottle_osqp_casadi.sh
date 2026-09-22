#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

exec "$CONTROL_ROOT/trajectory/osqp_waypoint_smoother/runtime/launch_right_bottle_box.sh" "$@"
