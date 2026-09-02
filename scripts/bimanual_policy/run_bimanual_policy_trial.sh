#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TELEOP_ROOT="${NERO_TELEOP_ROOT:-/home/dev/nero_neo_teleop}"
cd "$ROOT"

PYTHON=/home/dev/enter/envs/lerobot/bin/python
LEFT_CAN="${PICO_LEFT_CAN_PORT:-can_left}"
RIGHT_CAN="${PICO_RIGHT_CAN_PORT:-can_right}"
LEFT_USB="${PICO_LEFT_CAN_USB_BUS:-1-2.3:1.0}"
RIGHT_USB="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"
WORLD_CAMERA="${NERO_WORLD_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.1:1.0-video-index0}"
LEFT_WRIST_CAMERA="${NERO_LEFT_WRIST_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.4-usb-0:1.3:1.0-video-index0}"
RIGHT_WRIST_CAMERA="${NERO_RIGHT_WRIST_CAMERA:-/dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.4:1.0-video-index0}"
POLICY_HOST="${NERO_POLICY_SERVER:-172.24.1.154}"
POLICY_PORT="${NERO_POLICY_PORT:-8000}"
SKIP_HOME="${NERO_POLICY_SKIP_HOME:-0}"
POLICY_WARMUP="${NERO_POLICY_WARMUP:-1}"
PROMPT="${NERO_POLICY_PROMPT:-grasp both sides of the towel from the staging position, fold it, and release it}"

execute=0
for arg in "$@"; do
  if [[ "$arg" == "--execute" ]]; then
    execute=1
  fi
done

"$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$LEFT_CAN" "$LEFT_USB"
"$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"
./ensure_bimanual_policy_server.sh

if [[ "$POLICY_WARMUP" == 1 ]]; then
  echo "[POLICY] warming normal and RTC inference paths without robot commands"
  "$PYTHON" "$ROOT/../diagnostics/bimanual/bimanual_policy_rtc_probe.py" \
    --policy-host "$POLICY_HOST" \
    --policy-port "$POLICY_PORT" \
    --execution-horizon 12 \
    --num-steps 3
fi

common=(
  --policy-host "$POLICY_HOST"
  --policy-port "$POLICY_PORT"
  --left-can "$LEFT_CAN"
  --right-can "$RIGHT_CAN"
  --world-camera "$WORLD_CAMERA"
  --left-wrist-camera "$LEFT_WRIST_CAMERA"
  --right-wrist-camera "$RIGHT_WRIST_CAMERA"
  --prompt "$PROMPT"
)

if [[ "$execute" == 0 ]]; then
  exec "$PYTHON" ./bimanual_guarded_policy_stream.py \
    "${common[@]}" --preflight-only "$@"
fi

if [[ "$SKIP_HOME" != 1 ]]; then
  echo "[TRIAL] returning both arms to demonstration Home before inference"
  "$TELEOP_ROOT/scripts/control/run_dual_home.sh" --execute
fi

set +e
"$PYTHON" ./bimanual_guarded_policy_stream.py \
  "${common[@]}" \
  --confirm 'RUN GUARDED BIMANUAL POLICY' \
  "$@"
status=$?
set -e

if [[ "$status" != 0 ]]; then
  echo "[TRIAL] policy failed (exit=${status}); both arms were held and automatic Home is disabled" >&2
  exit "$status"
fi

if [[ "$SKIP_HOME" != 1 ]]; then
  echo "[TRIAL] successful policy trial; returning both arms to demonstration Home"
  "$TELEOP_ROOT/scripts/control/run_dual_home.sh" --execute
fi
