#!/usr/bin/env bash
set -euo pipefail

# Execute the learned two-stage towel workflow with an explicit, observable
# boundary between the right-arm staging action and the bimanual fold action.
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL_ROOT="$(cd "$ROOT/../.." && pwd)"
STAGE1_DURATION="${NERO_STAGE1_DURATION:-60}"
STAGE23_DURATION="${NERO_STAGE23_DURATION:-60}"
CHUNK_MODE="${NERO_POLICY_CHUNK_MODE:-feedback_phase}"
PYTHON=/home/dev/enter/envs/lerobot/bin/python
POLICY_LOG_ROOT="$CONTROL_ROOT/artifacts/logs/bimanual_policy_stream"

JOINT_POLICY_CONFIG="${NERO_TOWEL_POLICY_CONFIG:-pi05_nero_towel_multistage_99_next_feedback_event4_h24_v1}"
JOINT_POLICY_CHECKPOINT="${NERO_TOWEL_POLICY_CHECKPOINT:-16000}"
JOINT_POLICY_SOURCE="${NERO_TOWEL_POLICY_SOURCE:-/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_multistage_99_next_feedback_event4_h24_v1/lora_20000_towel_multistage_99_next_feedback_event4_h24_v1/${JOINT_POLICY_CHECKPOINT}}"
JOINT_POLICY_STAGE_NAME="${NERO_TOWEL_POLICY_STAGE_NAME:-nero_towel_multistage_next_feedback_event4_h24_${JOINT_POLICY_CHECKPOINT}}"

stage1_prompt="grasp the middle of the towel and place it at the staging position"
stage23_prompt="grasp both sides of the towel from the staging position, fold it, and release it"

echo "[WORKFLOW] stage 1: right-arm reposition; waiting for close -> release -> settle"
NERO_POLICY_CONFIG="$JOINT_POLICY_CONFIG" \
NERO_POLICY_CHECKPOINT="$JOINT_POLICY_CHECKPOINT" \
NERO_POLICY_SOURCE="$JOINT_POLICY_SOURCE" \
NERO_POLICY_STAGE_NAME="$JOINT_POLICY_STAGE_NAME" \
NERO_POLICY_PROMPT="$stage1_prompt" "$ROOT/run_bimanual_policy_trial.sh" \
  --duration "$STAGE1_DURATION" \
  --chunk-mode "$CHUNK_MODE" \
  --exit-on-right-gripper-cycle \
  --execute \
  "$@"

stage1_summary="$(find "$POLICY_LOG_ROOT" -name summary.json -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$stage1_summary" ]]; then
  echo "[FAIL] stage 1 produced no policy summary; refusing stage 2/3" >&2
  exit 1
fi
"$PYTHON" - "$stage1_summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
reason = summary.get("completion_reason")
if reason != "right_gripper_cycle_settled":
    raise SystemExit(
        "[FAIL] stage 1 did not reach the grasp/release/settle boundary "
        f"(completion_reason={reason!r}); refusing stage 2/3"
    )
print("[WORKFLOW] stage 1 boundary confirmed")
PY

echo "[WORKFLOW] stage 1 complete; Home return finished by the trial launcher"
echo "[WORKFLOW] stage 2/3: bimanual grasp, fold, and release"
NERO_POLICY_CONFIG="$JOINT_POLICY_CONFIG" \
NERO_POLICY_CHECKPOINT="$JOINT_POLICY_CHECKPOINT" \
NERO_POLICY_SOURCE="$JOINT_POLICY_SOURCE" \
NERO_POLICY_STAGE_NAME="$JOINT_POLICY_STAGE_NAME" \
NERO_POLICY_PROMPT="$stage23_prompt" "$ROOT/run_bimanual_policy_trial.sh" \
  --duration "$STAGE23_DURATION" \
  --chunk-mode "$CHUNK_MODE" \
  --execute \
  "$@"
