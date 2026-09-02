#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

export NERO_POLICY_CONFIG=pi05_nero_towel_bimanual_50_v1
export NERO_POLICY_CHECKPOINT=19999
export NERO_POLICY_SOURCE=/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_bimanual_50_v1/lora_20000_towel_bimanual_50_20260817/19999
export NERO_POLICY_STAGE_NAME=nero_towel_fullflow_20260817_h16_19999
export NERO_POLICY_PROMPT="fold the towel"
# This wrapper performs its own horizon-16 normal and RTC warmup.
export NERO_POLICY_WARMUP=0

"$ROOT/ensure_bimanual_policy_server.sh"
/home/dev/enter/envs/lerobot/bin/python \
  "$ROOT/../diagnostics/bimanual/bimanual_policy_rtc_probe.py" \
  --policy-host "${NERO_POLICY_SERVER:-172.24.1.154}" \
  --policy-port "${NERO_POLICY_PORT:-8000}" \
  --action-horizon 16 \
  --execution-horizon 12 \
  --num-steps 3

exec "$ROOT/run_bimanual_policy_trial.sh" \
  --action-horizon 16 \
  --chunk-mode feedback_phase \
  --duration 35 \
  "$@"
