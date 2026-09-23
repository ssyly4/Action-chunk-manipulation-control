#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTION="${1:-}"

if [[ -f "$ROOT/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/config/paths.env"
fi

case "$ACTION" in
  start)
    shift
    exec "$ROOT/scripts/run_policy.sh" --server-only "$@"
    ;;
  status|stop)
    shift
    [[ $# -eq 0 ]] || { echo "usage: $0 $ACTION" >&2; exit 2; }
    exec "$ROOT/scripts/bimanual_policy/ensure_bimanual_policy_server.sh" "$ACTION"
    ;;
  *)
    echo "usage: $0 {start|status|stop} [policy selection]" >&2
    exit 2
    ;;
esac
