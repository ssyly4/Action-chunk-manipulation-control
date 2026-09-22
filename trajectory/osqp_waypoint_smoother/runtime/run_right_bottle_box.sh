#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL_ROOT="$(cd "$ROOT/../../.." && pwd)"
TRAJECTORY_ROOT="$CONTROL_ROOT/trajectory"
CASADI_ROOT="$TRAJECTORY_ROOT/casadi_fixed_horizon_retimer"
TOPPRA_ROOT="$TRAJECTORY_ROOT/toppra_fixed_horizon_retimer"
TELEOP_ROOT="${NERO_TELEOP_ROOT:-/home/dev/nero_neo_teleop}"
ARM_SDK_ROOT="${NERO_ARM_SDK_ROOT:-/home/dev/nero_ws/src/pyAgxArm}"
PYTHON=/home/dev/enter/envs/lerobot/bin/python
RIGHT_CAN="${PICO_RIGHT_CAN_PORT:-can_right}"
RIGHT_USB="${PICO_RIGHT_CAN_USB_BUS:-1-2.3:1.0}"
WORLD_CAMERA="${NERO_WORLD_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.1:1.0-video-index0}"
RIGHT_WRIST_CAMERA="${NERO_RIGHT_WRIST_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.4:1.0-video-index0}"
POLICY_HOST="${NERO_POLICY_SERVER:-172.24.1.154}"
POLICY_PORT="${NERO_POLICY_PORT:-8000}"
PROMPT="${NERO_POLICY_PROMPT:-pick up the bottle and place it into the box}"
START_POSE="${NERO_RIGHT_START_POSE_FILE:-/home/dev/jepa_world_model_real_robot/configs/nero_bottle_into_box_right_start.local.json}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-osqp-casadi-retimer-right
RUNTIME_PYTHONPATH="$ROOT/../vendor:$ROOT/..:$ROOT:$CASADI_ROOT/vendor:$CASADI_ROOT:$CASADI_ROOT/runtime:$TOPPRA_ROOT/vendor:$TOPPRA_ROOT:$TOPPRA_ROOT/runtime:$CONTROL_ROOT:$ARM_SDK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

execute=0
for arg in "$@"; do
  [[ "$arg" == --execute ]] && execute=1
done

if [[ "${NERO_POLICY_WARMUP:-1}" == 1 ]]; then
  echo "[POLICY] warming normal and RTC paths without robot commands"
  env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" \
    "$CONTROL_ROOT/scripts/diagnostics/right_policy_rtc_warmup.py" \
    --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" \
    --action-horizon 16 --execution-horizon 8 --inference-delay 7 \
    --num-steps 3 --max-guidance-weight 2.0 --prompt "$PROMPT"
fi

"$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"

common=(
  --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT"
  --left-can virtual_left --right-can "$RIGHT_CAN"
  --world-camera "$WORLD_CAMERA"
  --left-wrist-camera virtual_left
  --right-wrist-camera "$RIGHT_WRIST_CAMERA"
  --prompt "$PROMPT"
)

if [[ "$execute" == 0 ]]; then
  exec env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" "$ROOT/right_policy_runtime.py" \
    "${common[@]}" --preflight-only "$@"
fi

if [[ ! -f "$START_POSE" ]]; then
  echo "[FAIL] right-arm demonstration start pose not found: $START_POSE" >&2
  exit 2
fi
RIGHT_HOME_DEG="$($PYTHON -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["start_joints_deg"]))' "$START_POSE")"

if [[ "${NERO_POLICY_SKIP_HOME:-0}" != 1 ]]; then
  echo "[TRIAL] returning right arm to the bottle demonstration start"
  env NERO_RIGHT_HOME_DEG="$RIGHT_HOME_DEG" PYTHONPATH="$TELEOP_ROOT/src:$CONTROL_ROOT:$ARM_SDK_ROOT" \
    "$PYTHON" -m nero_neo_teleop.robot.single_home \
    --can-port "$RIGHT_CAN" --home-side right --speed-percent 5 --execute \
    --confirm 'MOVE NERO ARM TO PICO HOME'
  "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
fi

set +e
env PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON" "$ROOT/right_policy_runtime.py" \
  "${common[@]}" --confirm 'RUN GUARDED BIMANUAL POLICY' "$@"
status=$?
set -e

if [[ "$status" != 0 ]]; then
  echo "[TRIAL] right OSQP+CasADi control failed (exit=$status); automatic Home is disabled" >&2
  exit "$status"
fi

if [[ "${NERO_POLICY_SKIP_HOME:-0}" != 1 ]]; then
  echo "[TRIAL] trial succeeded; returning right arm to the bottle demonstration start"
  "$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
  env NERO_RIGHT_HOME_DEG="$RIGHT_HOME_DEG" PYTHONPATH="$TELEOP_ROOT/src:$CONTROL_ROOT:$ARM_SDK_ROOT" \
    "$PYTHON" -m nero_neo_teleop.robot.single_home \
    --can-port "$RIGHT_CAN" --home-side right --speed-percent 5 --execute \
    --confirm 'MOVE NERO ARM TO PICO HOME'
fi
