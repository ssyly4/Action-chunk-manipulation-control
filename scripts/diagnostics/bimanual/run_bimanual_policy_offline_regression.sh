#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/dev/enter/envs/lerobot/bin/python

export PYTHONPATH="/home/dev/nero_ws:/home/dev/nero_ws/src/pyAgxArm:${PYTHONPATH:-}"

exec "$PYTHON" "$ROOT/bimanual_policy_offline_regression.py" "$@"
