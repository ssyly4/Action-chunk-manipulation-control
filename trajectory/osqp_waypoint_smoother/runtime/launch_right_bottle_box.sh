#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_bottle_box_right60_command_eff4_v1
export NERO_POLICY_CHECKPOINT=119999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_bottle_box_right60_command_eff4_v1/lora_opt30000_bottle_box_right60_command_h16_eff4_v1/119999
export NERO_POLICY_STAGE_NAME=nero_bottle_box_right60_h16_30000_osqp_casadi
export NERO_POLICY_PROMPT='pick up the bottle and place it into the box'

export PICO_RIGHT_CAN_USB_BUS="${PICO_RIGHT_CAN_USB_BUS:-1-2.3:1.0}"
export NERO_OSQP_FAST_PATH="${NERO_OSQP_FAST_PATH:-1}"
export NERO_OSQP_SPECULATIVE_RECOVERY="${NERO_OSQP_SPECULATIVE_RECOVERY:-0}"
export NERO_CASADI_PREWARM="${NERO_CASADI_PREWARM:-1}"
export NERO_CASADI_PREWARM_HORIZON="${NERO_CASADI_PREWARM_HORIZON:-16}"
export NERO_CASADI_PREWARM_PHASES="${NERO_CASADI_PREWARM_PHASES:-5,6,7}"
export NERO_CASADI_MAX_JERK_DEG_S3="${NERO_CASADI_MAX_JERK_DEG_S3:-8000}"
export NERO_CASADI_STREAMING_POSITION_GAIN_S="${NERO_CASADI_STREAMING_POSITION_GAIN_S:-4.0}"
export NERO_CASADI_MAX_SOLVE_SEC="${NERO_CASADI_MAX_SOLVE_SEC:-0.02}"
export NERO_OSQP_TRUST_REGION_DEG="${NERO_OSQP_TRUST_REGION_DEG:-0.3}"
export NERO_OSQP_MAX_SOLVE_SEC="${NERO_OSQP_MAX_SOLVE_SEC:-0.02}"
export NERO_TOPPRA_MAX_VELOCITY_DEG_S="${NERO_FOLLOWER_MAX_VELOCITY_DEG_S:-28}"
export NERO_TOPPRA_MAX_ACCELERATION_DEG_S2="${NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2:-280}"
export NERO_TOPPRA_RUNTIME_LOG="${NERO_OSQP_CASADI_RUNTIME_LOG:-$ROOT/../outputs/right_bottle_$(date +%Y%m%d_%H%M%S).jsonl}"

exec "$ROOT/run_right_bottle_box.sh" \
  --action-horizon 16 --duration "${NERO_POLICY_DURATION:-20}" \
  --chunk-mode rtc_time --rtc-execution-horizon 8 --rtc-queue-threshold 14 \
  --rtc-action-hz 30 --rtc-inference-delay-steps "${NERO_RTC_INFERENCE_DELAY_STEPS:-7}" \
  --rtc-handoff-decay-steps 0 --rtc-max-handoff-error-deg 2.5 \
  --rtc-num-steps 3 --rtc-max-guidance-weight 2.0 \
  --max-velocity-deg-s "${NERO_FOLLOWER_MAX_VELOCITY_DEG_S:-28}" \
  --max-acceleration-deg-s2 "${NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2:-280}" \
  --max-command-error-deg "${NERO_FOLLOWER_HARD_ERROR_DEG:-2.25}" \
  --feedback-governor-error-deg "${NERO_FOLLOWER_GOVERNOR_ERROR_DEG:-1.5}" \
  --progress-gripper-lead-steps 4 --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 0 --exit-on-right-gripper-cycle \
  "$@"
