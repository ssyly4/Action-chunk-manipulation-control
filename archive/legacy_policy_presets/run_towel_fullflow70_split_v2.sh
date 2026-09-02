#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OPTIMIZER_STEP=20000
forward_args=()

while (($#)); do
  case "$1" in
    --checkpoint)
      if (($# < 2)); then
        echo "error: --checkpoint requires an optimizer step such as 18K or 20000" >&2
        exit 2
      fi
      value="${2%[Kk]}"
      if [[ "$2" == *[Kk] ]]; then
        OPTIMIZER_STEP=$((10#$value * 1000))
      else
        OPTIMIZER_STEP=$((10#$value))
      fi
      shift 2
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

case "$OPTIMIZER_STEP" in
  12000) MICRO_STEP=48000 ;;
  13000) MICRO_STEP=52000 ;;
  14000) MICRO_STEP=56000 ;;
  15000) MICRO_STEP=60000 ;;
  16000) MICRO_STEP=64000 ;;
  17000) MICRO_STEP=68000 ;;
  18000) MICRO_STEP=72000 ;;
  19000) MICRO_STEP=76000 ;;
  20000) MICRO_STEP=80000 ;;
  24000) MICRO_STEP=95999 ;;
  *)
    echo "error: checkpoint must be one of 12K, 13K, 14K, 15K, 16K, 17K, 18K, 19K, 20K, or 24K" >&2
    exit 2
    ;;
esac

CONFIG=pi05_nero_towel_fullflow_70_next_feedback_event4_h24_split_v2
EXP=lora_micro96000_towel_fullflow70_next_feedback_h24_eff4_v2

export NERO_POLICY_CONFIG="$CONFIG"
export NERO_POLICY_CHECKPOINT="$MICRO_STEP"
export NERO_POLICY_SOURCE="/home/dev/workspace/nero_training/checkpoints/${CONFIG}/${EXP}/${MICRO_STEP}"
export NERO_POLICY_STAGE_NAME="nero_towel_fullflow70_split_v2_opt${OPTIMIZER_STEP}_micro${MICRO_STEP}"
export NERO_POLICY_PROMPT='fold the towel'
export NERO_POLICY_WARMUP=1

echo "[MODEL] split-v2 optimizer_step=${OPTIMIZER_STEP} checkpoint_dir=${MICRO_STEP}"

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 24 \
  --duration 50 \
  --chunk-mode rtc \
  --rtc-execution-horizon 12 \
  --rtc-queue-threshold 22 \
  --rtc-action-hz 25 \
  --rtc-num-steps 3 \
  --rtc-max-guidance-weight 1.0 \
  --max-velocity-deg-s 18 \
  --max-acceleration-deg-s2 120 \
  --max-alignment-error-deg 3.0 \
  --alignment-search-margin-steps 5.0 \
  --max-command-error-deg 1.75 \
  --feedback-governor-error-deg 1.0 \
  --progress-gripper-lead-steps 4 \
  --gripper-event-lookahead-steps 15 \
  --gripper-catchup-hold-sec 2.0 \
  --fail-on-gripper-catchup-timeout \
  --right-pregrasp-descent-mm 0 \
  --execute \
  "${forward_args[@]}"
