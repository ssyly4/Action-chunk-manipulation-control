#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=119999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1/lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1/119999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_releasecrop_tailpush30_h24_30000_osqp_casadi
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP="${NERO_POLICY_WARMUP:-1}"

export PICO_LEFT_CAN_USB_BUS="${PICO_LEFT_CAN_USB_BUS:-1-2.2:1.0}"
export PICO_RIGHT_CAN_USB_BUS="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"
export NERO_TOPPRA_BOUNDARY_PATH_MODE=path_curvature
export NERO_TOPPRA_CURVATURE_SPEED_MARGIN="${NERO_TOPPRA_CURVATURE_SPEED_MARGIN:-0.80}"

FOLLOWER_MAX_VELOCITY_DEG_S="${NERO_FOLLOWER_MAX_VELOCITY_DEG_S:-28}"
FOLLOWER_MAX_ACCELERATION_DEG_S2="${NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2:-280}"
FOLLOWER_GOVERNOR_ERROR_DEG="${NERO_FOLLOWER_GOVERNOR_ERROR_DEG:-1.50}"
FOLLOWER_HARD_ERROR_DEG="${NERO_FOLLOWER_HARD_ERROR_DEG:-2.25}"
export NERO_TOPPRA_MAX_VELOCITY_DEG_S="$FOLLOWER_MAX_VELOCITY_DEG_S"
export NERO_TOPPRA_MAX_ACCELERATION_DEG_S2="$FOLLOWER_MAX_ACCELERATION_DEG_S2"
export NERO_CASADI_MAX_JERK_DEG_S3="${NERO_CASADI_MAX_JERK_DEG_S3:-8000}"
export NERO_CASADI_STREAMING_POSITION_GAIN_S="${NERO_CASADI_STREAMING_POSITION_GAIN_S:-4.0}"
export NERO_CASADI_MAX_SOLVE_SEC="${NERO_CASADI_MAX_SOLVE_SEC:-0.20}"
export NERO_OSQP_TRUST_REGION_DEG="${NERO_OSQP_TRUST_REGION_DEG:-0.3}"
export NERO_OSQP_MAX_SOLVE_SEC="${NERO_OSQP_MAX_SOLVE_SEC:-0.02}"
export NERO_OSQP_ENFORCE_BOUNDARY="${NERO_OSQP_ENFORCE_BOUNDARY:-0}"
export NERO_TOPPRA_RUNTIME_LOG="${NERO_OSQP_CASADI_RUNTIME_LOG:-$ROOT/../outputs/runtime_$(date +%Y%m%d_%H%M%S).jsonl}"

export NERO_TOPPRA_MIN_ACTION_GAIN="${NERO_TOPPRA_MIN_ACTION_GAIN:-0.50}"
export NERO_TOPPRA_ACTION_GAIN_STEP="${NERO_TOPPRA_ACTION_GAIN_STEP:-0.025}"
export NERO_TOPPRA_MAX_ACTION_GAIN_RISE="${NERO_TOPPRA_MAX_ACTION_GAIN_RISE:-0.10}"
export NERO_TOPPRA_MIN_BLEND_TICKS="${NERO_TOPPRA_MIN_BLEND_TICKS:-2}"
export NERO_TOPPRA_MAX_BLEND_TICKS="${NERO_TOPPRA_MAX_BLEND_TICKS:-12}"
export NERO_TOPPRA_MIN_COMMIT_TICKS="${NERO_TOPPRA_MIN_COMMIT_TICKS:-8}"
export NERO_TOPPRA_PENDING_REQUEST_BLOCK_STEPS="${NERO_TOPPRA_PENDING_REQUEST_BLOCK_STEPS:-23}"
export NERO_TOPPRA_EMERGENCY_RESERVE_TICKS="${NERO_TOPPRA_EMERGENCY_RESERVE_TICKS:-3}"
export NERO_TOPPRA_REPLAN_RESERVE_TICKS="${NERO_TOPPRA_REPLAN_RESERVE_TICKS:-8}"
export NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF="${NERO_TOPPRA_ALLOW_RESERVE_FOLLOWER_HANDOFF:-0}"

echo "[FOLLOWER] feedback governor=${FOLLOWER_GOVERNOR_ERROR_DEG}deg hard_error=${FOLLOWER_HARD_ERROR_DEG}deg"

mode_args=(--execute)
for arg in "$@"; do
  if [[ "$arg" == --preflight-only ]]; then
    mode_args=()
    break
  fi
done

exec "$ROOT/run_policy.sh" \
  --action-horizon 24 --duration "${NERO_POLICY_DURATION:-30}" \
  --chunk-mode rtc_time --rtc-execution-horizon 12 --rtc-queue-threshold 22 \
  --rtc-action-hz 30 --rtc-handoff-decay-steps 0 \
  --rtc-max-handoff-error-deg 2.5 --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s "$FOLLOWER_MAX_VELOCITY_DEG_S" \
  --max-acceleration-deg-s2 "$FOLLOWER_MAX_ACCELERATION_DEG_S2" \
  --max-command-error-deg "$FOLLOWER_HARD_ERROR_DEG" \
  --feedback-governor-error-deg "$FOLLOWER_GOVERNOR_ERROR_DEG" \
  --progress-gripper-lead-steps 4 --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 0 "${mode_args[@]}" "$@"
