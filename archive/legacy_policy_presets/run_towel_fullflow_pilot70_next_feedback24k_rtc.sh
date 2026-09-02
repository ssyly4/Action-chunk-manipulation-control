#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Best checkpoint selected by the held-out validation ranking: 95,999
# microsteps, equivalent to approximately 24,000 optimizer updates.
export NERO_POLICY_CONFIG=pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3
export NERO_POLICY_CHECKPOINT=95999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3/lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3/95999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow_pilot70_next_feedback_h24_split_v3_95999_rtc
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP=1
# Current physical CAN adapter locations. The generic runner defaults are
# intentionally not used because they reflect the previous hub port layout.
export PICO_LEFT_CAN_USB_BUS=1-2.2:1.0
export PICO_RIGHT_CAN_USB_BUS=3-1.2:1.0
# Post-release height fallback only. The model retains control of normal
# vertical motion; no fixed-height locking or Cartesian push is applied.
POST_RELEASE_FLOOR_MM="${NERO_POST_RELEASE_FLOOR_MM:-138}"
POST_RELEASE_RECOVERY_MM="${NERO_POST_RELEASE_RECOVERY_MM:-143}"

# RTC generates overlapping action windows. The executor consumes them on a
# 30 Hz clock and skips only rows elapsed while the next inference ran.
exec "$ROOT/run_bimanual_policy_trial.sh" \
  --duration 30 \
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
  --post-release-recovery-mm "$POST_RELEASE_RECOVERY_MM" \
  --post-release-height-timeout-sec 6 \
  --execute \
  "$@"
