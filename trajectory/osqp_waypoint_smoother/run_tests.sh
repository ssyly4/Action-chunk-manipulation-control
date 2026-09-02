#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${NERO_OSQP_PYTHON:-/home/dev/enter/envs/lerobot/bin/python}"
CASADI_ROOT="$ROOT_DIR/../casadi_fixed_horizon_retimer"
TOPPRA_ROOT="$ROOT_DIR/../toppra_fixed_horizon_retimer"

PYTHONPATH="$ROOT_DIR/vendor:$ROOT_DIR:$ROOT_DIR/ab_runtime:$CASADI_ROOT/vendor:$CASADI_ROOT:$CASADI_ROOT/ab_runtime:$TOPPRA_ROOT/vendor:$TOPPRA_ROOT:$TOPPRA_ROOT/ab_runtime${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m unittest discover -s "$ROOT_DIR/tests" -q
