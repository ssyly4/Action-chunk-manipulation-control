#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# 119999 microsteps / microstep-4 = 30000 optimizer steps.
export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=119999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1/lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1/119999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_releasecrop_tailpush30_h24_30000_toppra_receding_ab
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP="${NERO_POLICY_WARMUP:-1}"

export PICO_LEFT_CAN_USB_BUS="${PICO_LEFT_CAN_USB_BUS:-1-2.2:1.0}"
export PICO_RIGHT_CAN_USB_BUS="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"

# TOPPRA and the production follower must describe the same physical envelope.
FOLLOWER_MAX_VELOCITY_DEG_S="${NERO_FOLLOWER_MAX_VELOCITY_DEG_S:-25}"
FOLLOWER_MAX_ACCELERATION_DEG_S2="${NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2:-220}"
export NERO_TOPPRA_MAX_VELOCITY_DEG_S="$FOLLOWER_MAX_VELOCITY_DEG_S"
export NERO_TOPPRA_MAX_ACCELERATION_DEG_S2="$FOLLOWER_MAX_ACCELERATION_DEG_S2"

# Reduce each new arm trajectory around live feedback instead of accumulating
# unreachable target debt. One gain is shared by all 14 arm joints; grippers
# remain unchanged.
export NERO_TOPPRA_MIN_ACTION_GAIN="${NERO_TOPPRA_MIN_ACTION_GAIN:-0.50}"
export NERO_TOPPRA_ACTION_GAIN_STEP="${NERO_TOPPRA_ACTION_GAIN_STEP:-0.025}"
export NERO_TOPPRA_MAX_ACTION_GAIN_RISE="${NERO_TOPPRA_MAX_ACTION_GAIN_RISE:-0.10}"
export NERO_TOPPRA_MIN_BLEND_TICKS="${NERO_TOPPRA_MIN_BLEND_TICKS:-2}"
export NERO_TOPPRA_MAX_BLEND_TICKS="${NERO_TOPPRA_MAX_BLEND_TICKS:-12}"

mode_args=(--execute)
for arg in "$@"; do
  if [[ "$arg" == --preflight-only ]]; then
    mode_args=()
    break
  fi
done

exec "$ROOT/run_policy_trial_toppra.sh" \
  --action-horizon 24 \
  --duration "${NERO_POLICY_DURATION:-15}" \
  --chunk-mode rtc_time \
  --rtc-execution-horizon 12 \
  --rtc-queue-threshold 22 \
  --rtc-action-hz 30 \
  --rtc-handoff-decay-steps 0 \
  --rtc-max-handoff-error-deg 2.5 \
  --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s "$FOLLOWER_MAX_VELOCITY_DEG_S" \
  --max-acceleration-deg-s2 "$FOLLOWER_MAX_ACCELERATION_DEG_S2" \
  --max-command-error-deg 2.25 \
  --feedback-governor-error-deg 1.25 \
  --progress-gripper-lead-steps 4 \
  --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 0 \
  "${mode_args[@]}" \
  "$@"
