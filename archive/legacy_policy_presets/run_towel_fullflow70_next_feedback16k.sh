#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=16000
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1/lora_24000_towel_fullflow_70_next_feedback_event4_h24_20260824/16000
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_next_feedback_event4_h24_16000
export NERO_POLICY_PROMPT='fold the towel'

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration 60 \
  --chunk-mode feedback_phase \
  --gripper-event-lookahead-steps 0 \
  --right-pregrasp-descent-mm 0 \
  --execute \
  "$@"
