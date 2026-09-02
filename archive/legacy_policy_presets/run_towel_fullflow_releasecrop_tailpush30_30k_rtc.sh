#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# 119,999 microsteps with microstep-4 training equals 30,000 optimizer steps.
export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1
export NERO_POLICY_CHECKPOINT=119999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1/lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1/119999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow70_releasecrop_tailpush30_h24_30000
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP=1

# Current physical adapter placement. Environment values override these when
# the CAN adapters are moved.
export PICO_LEFT_CAN_USB_BUS="${PICO_LEFT_CAN_USB_BUS:-1-2.2:1.0}"
export PICO_RIGHT_CAN_USB_BUS="${PICO_RIGHT_CAN_USB_BUS:-3-1.2:1.0}"

# Once the right gripper has completed its close/open cycle, first recover to
# the working plane, then allow the learned sweep to descend only to this TCP
# floor. This prevents the post-release policy tail from pressing the towel.
POST_RELEASE_FLOOR_MM="${NERO_POST_RELEASE_FLOOR_MM:-145}"
POST_RELEASE_WORK_HEIGHT_MM="${NERO_POST_RELEASE_WORK_HEIGHT_MM:-170}"

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration "${NERO_POLICY_DURATION:-30}" \
  --chunk-mode rtc_time \
  --rtc-execution-horizon 12 \
  --rtc-queue-threshold 22 \
  --rtc-action-hz 30 \
  --rtc-handoff-decay-steps 6 \
  --rtc-max-handoff-error-deg 2.5 \
  --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s 20 \
  --max-acceleration-deg-s2 160 \
  --max-command-error-deg 2.25 \
  --feedback-governor-error-deg 1.25 \
  --progress-gripper-lead-steps 4 \
  --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 2.0 \
  --fail-on-gripper-catchup-timeout \
  --right-pregrasp-descent-mm 0 \
  --right-post-release-height-guard \
  --post-release-floor-mm "$POST_RELEASE_FLOOR_MM" \
  --post-release-recovery-mm "$POST_RELEASE_WORK_HEIGHT_MM" \
  --post-release-height-timeout-sec 6 \
  --post-release-rtc-max-right-handoff-error-deg 6 \
  --execute \
  "$@"
