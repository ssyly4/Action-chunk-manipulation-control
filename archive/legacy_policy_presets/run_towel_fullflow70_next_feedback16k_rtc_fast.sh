#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=16000
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1/lora_24000_towel_fullflow_70_next_feedback_event4_h24_20260824/16000
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_next_feedback_event4_h24_16000_rtc_fast
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP=1

# The longest training demonstration is 19.23 s. This policy has no learned
# done signal, so a longer rollout repeatedly extrapolates beyond its data.
exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration 20 \
  --chunk-mode rtc \
  --rtc-execution-horizon 12 \
  --rtc-queue-threshold 22 \
  --rtc-action-hz 25 \
  --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s 14 \
  --max-acceleration-deg-s2 80 \
  --max-command-error-deg 1.75 \
  --feedback-governor-error-deg 1.0 \
  --progress-gripper-lead-steps 4 \
  --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 2.0 \
  --fail-on-gripper-catchup-timeout \
  --right-pregrasp-descent-mm 0 \
  --execute \
  "$@"
