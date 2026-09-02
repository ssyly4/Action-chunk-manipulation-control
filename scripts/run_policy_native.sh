#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_PATHS="$CONTROL_ROOT/config/paths.env"
if [[ -f "$LOCAL_PATHS" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_PATHS"
fi

exec "$CONTROL_ROOT/scripts/bimanual_policy/run_bimanual_policy_trial.sh" "$@"
