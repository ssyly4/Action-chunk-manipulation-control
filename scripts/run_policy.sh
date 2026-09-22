#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME=towel_fold
SHOW_CONFIG=0
EXECUTE=0
PREFLIGHT=0
PASSTHROUGH=()

usage() {
  cat <<'EOF'
usage: ./scripts/run_policy.sh [--task NAME] [--show-config] [controller options]

tasks:
  towel_fold          dual-arm towel policy (default)
  bottle_to_box_right single-right-arm bottle policy

--show-config prints the resolved task without touching CAN, cameras, or robots.
Use --preflight-only for full hardware/policy validation without robot commands.
EOF
}

while (($#)); do
  case "$1" in
    --task)
      shift
      [[ $# -gt 0 ]] || { echo "[FAIL] --task requires a value" >&2; exit 2; }
      TASK_NAME="$1"
      ;;
    --task=*) TASK_NAME="${1#*=}" ;;
    --show-config) SHOW_CONFIG=1 ;;
    --execute) EXECUTE=1 ;;
    --preflight-only) PREFLIGHT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) PASSTHROUGH+=("$1") ;;
  esac
  shift
done

if [[ "$EXECUTE" == 1 && "$PREFLIGHT" == 1 ]]; then
  echo "[FAIL] --execute and --preflight-only are mutually exclusive" >&2
  exit 2
fi

LOCAL_PATHS="$CONTROL_ROOT/config/paths.env"
if [[ -f "$LOCAL_PATHS" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_PATHS"
fi

TASK_FILE="$CONTROL_ROOT/config/tasks/${TASK_NAME}.toml"
[[ -f "$TASK_FILE" ]] || { echo "[FAIL] unknown task: $TASK_NAME" >&2; usage >&2; exit 2; }
eval "$(PYTHONPATH="$CONTROL_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m nero_vla.task_config "$TASK_FILE")"

if [[ "$SHOW_CONFIG" == 1 ]]; then
  printf 'task=%s mode=%s runtime=%s\n' "$NERO_TASK_NAME" "$NERO_TASK_MODE" "$NERO_TASK_RUNTIME"
  printf 'policy=%s checkpoint=%s prompt=%q\n' "$NERO_POLICY_CONFIG" "$NERO_POLICY_CHECKPOINT" "$NERO_POLICY_PROMPT"
  printf 'horizon_args='; printf ' %q' "${NERO_TASK_ARGS[@]}"; printf '\n'
  exit 0
fi

TRAJECTORY_ROOT="$CONTROL_ROOT/trajectory"
OSQP_ROOT="$TRAJECTORY_ROOT/osqp_waypoint_smoother"
CASADI_ROOT="$TRAJECTORY_ROOT/casadi_fixed_horizon_retimer"
TOPPRA_ROOT="$TRAJECTORY_ROOT/toppra_fixed_horizon_retimer"
POLICY_ROOT="$CONTROL_ROOT/scripts/bimanual_policy"
TELEOP_ROOT="${NERO_TELEOP_ROOT:-/home/dev/nero_neo_teleop}"
ARM_SDK_ROOT="${NERO_ARM_SDK_ROOT:-/home/dev/nero_ws/src/pyAgxArm}"
PYTHON="${NERO_LEROBOT_PYTHON:-/home/dev/enter/envs/lerobot/bin/python}"
POLICY_HOST="${NERO_POLICY_SERVER:-172.24.1.154}"
POLICY_PORT="${NERO_POLICY_PORT:-8000}"
WORLD_CAMERA="${NERO_WORLD_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.1:1.0-video-index0}"
RUNTIME="$CONTROL_ROOT/$NERO_TASK_RUNTIME"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/matplotlib-nero-${NERO_TASK_NAME}"
export NERO_TOPPRA_MAX_VELOCITY_DEG_S="$NERO_FOLLOWER_MAX_VELOCITY_DEG_S"
export NERO_TOPPRA_MAX_ACCELERATION_DEG_S2="$NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2"
export NERO_TOPPRA_RUNTIME_LOG="${NERO_OSQP_CASADI_RUNTIME_LOG:-$OSQP_ROOT/outputs/${NERO_TASK_NAME}_$(date +%Y%m%d_%H%M%S).jsonl}"
RUNTIME_PYTHONPATH="$OSQP_ROOT/vendor:$OSQP_ROOT:$OSQP_ROOT/runtime:$CASADI_ROOT/vendor:$CASADI_ROOT:$CASADI_ROOT/runtime:$TOPPRA_ROOT/vendor:$TOPPRA_ROOT:$TOPPRA_ROOT/runtime:$CONTROL_ROOT:$ARM_SDK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE_ARGS=()
if [[ "$EXECUTE" == 1 ]]; then
  MODE_ARGS=(--execute)
fi

echo "[TASK] $NERO_TASK_NAME mode=$NERO_TASK_MODE"
echo "[FOLLOWER] velocity=${NERO_FOLLOWER_MAX_VELOCITY_DEG_S}deg/s acceleration=${NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2}deg/s2 governor=${NERO_FOLLOWER_GOVERNOR_ERROR_DEG}deg hard_error=${NERO_FOLLOWER_HARD_ERROR_DEG}deg"
echo "[RETIME] osqp_fast_path=${NERO_OSQP_FAST_PATH} speculative_recovery=${NERO_OSQP_SPECULATIVE_RECOVERY} casadi_budget=${NERO_CASADI_MAX_SOLVE_SEC}s"
echo "[RTC] fixed_delay=${NERO_RTC_INFERENCE_DELAY_STEPS}tick guidance=${NERO_RTC_MAX_GUIDANCE_WEIGHT}"

LEFT_CAN="${PICO_LEFT_CAN_PORT:-can_left}"
RIGHT_CAN="${PICO_RIGHT_CAN_PORT:-can_right}"
LEFT_USB="${PICO_LEFT_CAN_USB_BUS:-1-2.2:1.0}"
RIGHT_USB="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"
LEFT_WRIST_CAMERA="${NERO_LEFT_WRIST_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.4-usb-0:1.3:1.0-video-index0}"
RIGHT_WRIST_CAMERA="${NERO_RIGHT_WRIST_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.4:1.0-video-index0}"

if [[ "$NERO_TASK_MODE" == dual ]]; then
  "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$LEFT_CAN" "$LEFT_USB"
  "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
  COMMON=(--policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" --left-can "$LEFT_CAN" --right-can "$RIGHT_CAN" --world-camera "$WORLD_CAMERA" --left-wrist-camera "$LEFT_WRIST_CAMERA" --right-wrist-camera "$RIGHT_WRIST_CAMERA" --prompt "$NERO_POLICY_PROMPT")
else
  "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
  COMMON=(--policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" --left-can virtual_left --right-can "$RIGHT_CAN" --world-camera "$WORLD_CAMERA" --left-wrist-camera virtual_left --right-wrist-camera "$RIGHT_WRIST_CAMERA" --prompt "$NERO_POLICY_PROMPT")
fi

"$POLICY_ROOT/ensure_bimanual_policy_server.sh"

if [[ "${NERO_POLICY_WARMUP:-1}" == 1 ]]; then
  echo "[POLICY] warming normal and RTC paths without robot commands"
  if [[ "$NERO_TASK_WARMUP" == bimanual ]]; then
    "$PYTHON" "$CONTROL_ROOT/scripts/diagnostics/bimanual/bimanual_policy_rtc_probe.py" --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" --execution-horizon 12 --num-steps 3
  else
    env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" "$CONTROL_ROOT/scripts/diagnostics/right_policy_rtc_warmup.py" --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" --action-horizon 16 --execution-horizon 8 --inference-delay "$NERO_RTC_INFERENCE_DELAY_STEPS" --num-steps 3 --max-guidance-weight "$NERO_RTC_MAX_GUIDANCE_WEIGHT" --prompt "$NERO_POLICY_PROMPT"
  fi
fi

CONTROL_ARGS=("${COMMON[@]}" "${NERO_TASK_ARGS[@]}" --duration "$NERO_POLICY_DURATION" --rtc-inference-delay-steps "$NERO_RTC_INFERENCE_DELAY_STEPS" --rtc-max-guidance-weight "$NERO_RTC_MAX_GUIDANCE_WEIGHT" --max-velocity-deg-s "$NERO_FOLLOWER_MAX_VELOCITY_DEG_S" --max-acceleration-deg-s2 "$NERO_FOLLOWER_MAX_ACCELERATION_DEG_S2" --max-command-error-deg "$NERO_FOLLOWER_HARD_ERROR_DEG" --feedback-governor-error-deg "$NERO_FOLLOWER_GOVERNOR_ERROR_DEG")

if [[ "$EXECUTE" != 1 ]]; then
  [[ "$PREFLIGHT" == 1 ]] || echo "[SAFE DEFAULT] --execute was not supplied; running preflight only"
  exec env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" "$RUNTIME" "${CONTROL_ARGS[@]}" --preflight-only "${PASSTHROUGH[@]}"
fi

if [[ "${NERO_POLICY_SKIP_HOME:-0}" != 1 ]]; then
  if [[ "$NERO_TASK_MODE" == dual ]]; then
    echo "[TRIAL] returning both arms to demonstration Home"
    "$TELEOP_ROOT/scripts/control/run_dual_home.sh" --execute
  else
    START_POSE="${NERO_RIGHT_START_POSE_FILE:-/home/dev/jepa_world_model_real_robot/configs/nero_bottle_into_box_right_start.local.json}"
    [[ -f "$START_POSE" ]] || { echo "[FAIL] right-arm start pose not found: $START_POSE" >&2; exit 2; }
    RIGHT_HOME_DEG="$($PYTHON -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["start_joints_deg"]))' "$START_POSE")"
    echo "[TRIAL] returning right arm to the demonstration start"
    env NERO_RIGHT_HOME_DEG="$RIGHT_HOME_DEG" PYTHONPATH="$TELEOP_ROOT/src:$CONTROL_ROOT:$ARM_SDK_ROOT" "$PYTHON" -m nero_neo_teleop.robot.single_home --can-port "$RIGHT_CAN" --home-side right --speed-percent 5 --execute --confirm 'MOVE NERO ARM TO PICO HOME'
    "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
  fi
fi

set +e
env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" "$RUNTIME" "${CONTROL_ARGS[@]}" --confirm 'RUN GUARDED BIMANUAL POLICY' "${MODE_ARGS[@]}" "${PASSTHROUGH[@]}"
status=$?
set -e
if [[ "$status" != 0 ]]; then
  echo "[TRIAL] $NERO_TASK_NAME failed (exit=$status); automatic Home is disabled" >&2
  exit "$status"
fi

if [[ "${NERO_POLICY_SKIP_HOME:-0}" != 1 ]]; then
  if [[ "$NERO_TASK_MODE" == dual ]]; then
    "$TELEOP_ROOT/scripts/control/run_dual_home.sh" --execute
  else
    "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
    env NERO_RIGHT_HOME_DEG="$RIGHT_HOME_DEG" PYTHONPATH="$TELEOP_ROOT/src:$CONTROL_ROOT:$ARM_SDK_ROOT" "$PYTHON" -m nero_neo_teleop.robot.single_home --can-port "$RIGHT_CAN" --home-side right --speed-percent 5 --execute --confirm 'MOVE NERO ARM TO PICO HOME'
  fi
fi
