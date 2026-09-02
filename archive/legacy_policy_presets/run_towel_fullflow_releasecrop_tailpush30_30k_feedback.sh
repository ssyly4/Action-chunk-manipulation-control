#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Same 30K checkpoint as the RTC presets; only the action consumer changes.
export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=119999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1/lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1/119999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_releasecrop_tailpush30_h24_30000_feedback
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP=1

export PICO_LEFT_CAN_USB_BUS="${PICO_LEFT_CAN_USB_BUS:-1-2.2:1.0}"
export PICO_RIGHT_CAN_USB_BUS="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"

# Feedback-phase comparison: each arm advances through the shared action
# chunk only when the measured dual-arm joints match its current phase.
# No Cartesian task assist or gripper-lag arm hold is enabled.
exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration "${NERO_POLICY_DURATION:-30}" \
  --chunk-mode feedback_phase \
  --max-velocity-deg-s 20 \
  --max-acceleration-deg-s2 160 \
  --max-command-error-deg 2.25 \
  --feedback-governor-error-deg 1.25 \
  --progress-arm-lead-steps 1 \
  --progress-gripper-lead-steps 0 \
  --gripper-event-lookahead-steps 0 \
  --gripper-catchup-hold-sec 0 \
  --right-pregrasp-descent-mm 0 \
  --execute \
  "$@"
