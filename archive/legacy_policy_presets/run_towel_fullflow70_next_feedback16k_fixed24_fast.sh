#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ "${NERO_ALLOW_UNSAFE_FIXED24_DIAGNOSTIC:-0}" != 1 ]]; then
  cat >&2 <<'EOF'
[BLOCKED] fixed24_fast is a failed diagnostic, not a robot-control preset.
The 2026-08-24 trial repeatedly drove both arms inward toward collision.
Set NERO_ALLOW_UNSAFE_FIXED24_DIAGNOSTIC=1 only for offline/preflight analysis.
EOF
  exit 2
fi

export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=16000
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1/lora_24000_towel_fullflow_70_next_feedback_event4_h24_20260824/16000
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_next_feedback_event4_h24_16000
export NERO_POLICY_PROMPT='fold the towel'

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration 45 \
  --chunk-mode fixed_horizon \
  --fixed-horizon-steps 24 \
  --max-velocity-deg-s 25 \
  --max-acceleration-deg-s2 220 \
  --max-alignment-error-deg 2.5 \
  --max-command-error-deg 2.0 \
  --feedback-governor-error-deg 1.25 \
  --gripper-event-lookahead-steps 0 \
  --right-pregrasp-descent-mm 0 \
  --execute \
  "$@"
