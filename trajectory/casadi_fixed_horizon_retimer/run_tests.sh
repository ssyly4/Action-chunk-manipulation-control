#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_ROOT="$(cd "$ROOT/../.." && pwd)"
PYTHONPATH="$ROOT/vendor:$ROOT:$ROOT/runtime:$ROOT/../toppra_fixed_horizon_retimer/vendor:$ROOT/../toppra_fixed_horizon_retimer:$ROOT/../toppra_fixed_horizon_retimer/runtime:$CONTROL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  /home/dev/enter/envs/lerobot/bin/python -m unittest discover -s "$ROOT/tests" -v
