#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow_70_command_event4_h24_v1
export NERO_POLICY_CHECKPOINT="${NERO_POLICY_CHECKPOINT:-16000}"
export NERO_POLICY_SOURCE="/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_70_command_event4_h24_v1/lora_40000_towel_fullflow_70_command_event4_h24_20260822/${NERO_POLICY_CHECKPOINT}"
export NERO_POLICY_STAGE_NAME="nero_towel_fullflow70_command_event4_h24_${NERO_POLICY_CHECKPOINT}"
export NERO_POLICY_PROMPT='fold the towel'

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --duration 60 \
  --chunk-mode feedback_phase \
  --gripper-event-lookahead-steps 0 \
  --right-pregrasp-descent-mm 0 \
  --execute "$@"
