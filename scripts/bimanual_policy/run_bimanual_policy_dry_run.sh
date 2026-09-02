#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL_ROOT="$(cd "$ROOT/../.." && pwd)"
TELEOP_ROOT="${NERO_TELEOP_ROOT:-/home/dev/nero_neo_teleop}"
ARM_SDK_ROOT="${NERO_ARM_SDK_ROOT:-/home/dev/nero_ws/src/pyAgxArm}"
PYTHON=/home/dev/enter/envs/lerobot/bin/python
LEFT_CAN="${PICO_LEFT_CAN_PORT:-can_left}"
RIGHT_CAN="${PICO_RIGHT_CAN_PORT:-can_right}"
LEFT_USB="${PICO_LEFT_CAN_USB_BUS:-1-2.3:1.0}"
RIGHT_USB="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"
PROMPT="${NERO_POLICY_PROMPT:-grasp both sides of the towel from the staging position, fold it, and release it}"

export PYTHONPATH="$TELEOP_ROOT/src:$CONTROL_ROOT:$ARM_SDK_ROOT:${PYTHONPATH:-}"

"$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$LEFT_CAN" "$LEFT_USB"
"$TELEOP_ROOT/scripts/can/ensure_can_interface.sh" "$RIGHT_CAN" "$RIGHT_USB"

export NERO_TELEOP_SRC="$TELEOP_ROOT/src"

exec "$PYTHON" "$ROOT/bimanual_policy_dry_run.py" \
  --policy-host 172.24.1.154 \
  --policy-port 8000 \
  --left-can "$LEFT_CAN" \
  --right-can "$RIGHT_CAN" \
  --world-camera /dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.1:1.0-video-index0 \
  --left-wrist-camera /dev/v4l/by-path/pci-0000:07:00.4-usb-0:1.3:1.0-video-index0 \
  --right-wrist-camera /dev/v4l/by-path/pci-0000:07:00.3-usb-0:2.4:1.0-video-index0 \
  --prompt "$PROMPT" \
  "$@"
