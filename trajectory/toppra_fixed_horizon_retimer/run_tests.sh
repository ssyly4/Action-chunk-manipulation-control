#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTROL_ROOT="$(cd "$ROOT/../.." && pwd)"
export PYTHONPATH="$ROOT/vendor:$ROOT:$ROOT/runtime:$CONTROL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-toppra-retimer
exec /home/dev/enter/envs/lerobot/bin/python -m unittest discover -s "$ROOT/tests" -v
