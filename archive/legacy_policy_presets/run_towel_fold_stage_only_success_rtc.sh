#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Historical fold-only policy that completed the bimanual towel fold trials.
export NERO_POLICY_CONFIG=pi05_nero_towel_fold_stage_50_h24_v1
export NERO_POLICY_CHECKPOINT=19999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fold_stage_50_h24_v1/lora_20000_towel_fold_stage_50_h24_20260819/19999
export NERO_POLICY_STAGE_NAME=nero_towel_fold_stage_only_h24_19999_success_rtc
export NERO_POLICY_PROMPT="fold the towel"
export NERO_POLICY_WARMUP=1

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration 40 \
  --chunk-mode rtc \
  --rtc-execution-horizon 12 \
  --rtc-queue-threshold 22 \
  --rtc-action-hz 25 \
  --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s 8 \
  --max-acceleration-deg-s2 24 \
  --max-gripper-consecutive 0.35 \
  --gripper-event-lookahead-steps 10 \
  "$@"
