#!/usr/bin/env bash
set -euo pipefail

# Sixty single-right-arm PICO demonstrations for bottle-to-box SFT.
# The existing single-arm recorder owns CAN, cameras, Home and the dataset.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RECORDER_ROOT="${NERO_RIGHT_RECORDER_ROOT:-$(cd "$VLA_ROOT/.." && pwd)/jepa_world_model_real_robot}"
RECORDER="$RECORDER_ROOT/scripts/record_nero_right_50.sh"

usage() {
  echo "usage: $0 [--execute]"
  echo "  Without --execute: print the planned single-right-arm collection only."
  echo "  With --execute: collect 60 saved PICO demonstrations via the existing recorder."
  echo "  Configure cameras/CAN in nero_neo_teleop/.env; data root via NERO_RIGHT_DATA_DIR."
}

execute=0
for argument in "$@"; do
  case "$argument" in
    --execute) execute=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[FAIL] unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

export NERO_RIGHT_EPISODES=60
export NERO_RIGHT_DATASET_NAME="${NERO_RIGHT_DATASET_NAME:-nero_bottle_into_box_right_60_command_v1}"
export NERO_RIGHT_TASK="${NERO_RIGHT_TASK:-pick up the bottle and place it in the box}"
export NERO_RIGHT_DATA_DIR="${NERO_RIGHT_DATA_DIR:-${NERO_DATA_ROOT:-$HOME/nero_data}/raw/bottle_to_box}"
export NERO_RIGHT_EPISODE_SECONDS="${NERO_RIGHT_EPISODE_SECONDS:-20}"
export NERO_RIGHT_CAPTURE_CURRENT_START=1
export NERO_RIGHT_START_POSE_FILE="${NERO_RIGHT_START_POSE_FILE:-$RECORDER_ROOT/configs/nero_bottle_into_box_right_start.local.json}"
export NERO_RIGHT_INACTIVITY_AUTO_STOP_SECONDS="${NERO_RIGHT_INACTIVITY_AUTO_STOP_SECONDS:-0.5}"
export NERO_RIGHT_ACTION_SOURCE=controller_command

echo "[right PICO] saved episodes=60 action_source=controller_command"
echo "[right PICO] task=$NERO_RIGHT_TASK"
echo "[right PICO] dataset=$NERO_RIGHT_DATA_DIR/$NERO_RIGHT_DATASET_NAME"
echo "[right PICO] fixed start pose: $NERO_RIGHT_START_POSE_FILE"

if [[ "$execute" != 1 ]]; then
  echo "[PREVIEW ONLY] no camera, CAN or robot process was started"
  exit 0
fi
if [[ ! -f "$RECORDER" ]]; then
  echo "[FAIL] single-right-arm recorder not found: $RECORDER" >&2
  exit 2
fi
if [[ ! -f "$NERO_RIGHT_START_POSE_FILE" ]]; then
  echo "[FAIL] fixed start pose not found: $NERO_RIGHT_START_POSE_FILE" >&2
  exit 2
fi

exec bash "$RECORDER" --execute
