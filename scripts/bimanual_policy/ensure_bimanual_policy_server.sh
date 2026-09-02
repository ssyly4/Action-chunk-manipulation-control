#!/usr/bin/env bash
set -euo pipefail

SERVER="${NERO_POLICY_SERVER:-172.24.1.154}"
PORT="${NERO_POLICY_PORT:-8000}"
CONTAINER="${NERO_POLICY_CONTAINER:-cuda12_8_torch_2_9_1_core}"
CONFIG="${NERO_POLICY_CONFIG:-pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1}"
CHECKPOINT="${NERO_POLICY_CHECKPOINT:-16000}"
SOURCE="${NERO_POLICY_SOURCE:-/home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_70_next_feedback_event4_h24_v1/lora_24000_towel_fullflow_70_next_feedback_event4_h24_20260824/${CHECKPOINT}}"
STAGE_NAME="${NERO_POLICY_STAGE_NAME:-nero_towel_fullflow70_next_feedback_event4_h24_${CHECKPOINT}}"
HOST_STAGE_ROOT="${NERO_POLICY_HOST_STAGE_ROOT:-/home/dev/workspace/nero_training/inference_staging}"
HOST_STAGE="${NERO_POLICY_HOST_STAGE:-${HOST_STAGE_ROOT}/${STAGE_NAME}}"
CONTAINER_STAGE_ROOT="${NERO_POLICY_CONTAINER_STAGE_ROOT:-/tmp/nero_policy_staging}"
CONTAINER_STAGE="${NERO_POLICY_CONTAINER_STAGE:-${CONTAINER_STAGE_ROOT}/${STAGE_NAME}}"
LOG="${NERO_POLICY_LOG:-/home/dev/workspace/nero_training/logs/${STAGE_NAME}_policy_server.log}"

port_open() {
  timeout 2 bash -c "</dev/tcp/${SERVER}/${PORT}" >/dev/null 2>&1
}

if port_open; then
  if ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 "dev@${SERVER}" \
    "pgrep -af '[s]cripts/serve_policy.py.*${CONFIG}.*${CONTAINER_STAGE}' >/dev/null"; then
    echo "[POLICY] ready: ${SERVER}:${PORT} config=${CONFIG} checkpoint=${CHECKPOINT}"
    exit 0
  fi
  if ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 "dev@${SERVER}" \
    "pgrep -af '[s]cripts/serve_policy.py.*--port ${PORT}' >/dev/null"; then
    echo "[POLICY] switching active OpenPI policy to config=${CONFIG} checkpoint=${CHECKPOINT}"
    ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 "dev@${SERVER}" \
      "docker exec '${CONTAINER}' pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true"
    for _ in $(seq 1 20); do
      if ! port_open; then
        break
      fi
      sleep 1
    done
  else
    echo "[POLICY] ERROR: port ${PORT} is occupied by a non-OpenPI process" >&2
    exit 1
  fi
fi

echo "[POLICY] staging and starting ${CONFIG} on ${SERVER}:${PORT}"
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 "dev@${SERVER}" bash -s -- \
  "$CONTAINER" "$SOURCE" "$HOST_STAGE" "$CONTAINER_STAGE" "$CONFIG" "$PORT" "$LOG" <<'REMOTE'
set -euo pipefail
container="$1"
source="$2"
host_stage="$3"
container_stage="$4"
config="$5"
port="$6"
log="$7"

if ! docker exec "$container" test -d "$container_stage/params"; then
  if [[ ! -d "$host_stage/params" ]]; then
    partial="${host_stage}.partial"
    rm -rf "$partial"
    mkdir -p "$partial"
    cp -a "$source/params" "$partial/params"
    if [[ -d "$source/assets" ]]; then
      cp -a "$source/assets" "$partial/assets"
    fi
    if [[ -f "$source/_CHECKPOINT_METADATA" ]]; then
      cp -a "$source/_CHECKPOINT_METADATA" "$partial/_CHECKPOINT_METADATA"
    fi
    rm -rf "$host_stage"
    mv "$partial" "$host_stage"
  fi
  host_stage_root="$(dirname "$host_stage")"
  find "$host_stage_root" -mindepth 1 -maxdepth 1 -type d \
    ! -name "$(basename "$host_stage")" -exec rm -rf -- {} +
  container_stage_root="$(dirname "$container_stage")"
  docker exec "$container" rm -rf "$container_stage_root"
  docker exec "$container" mkdir -p "$container_stage_root"
  docker cp "$host_stage" "$container:$container_stage"
fi

docker exec --user root "$container" bash -lc \
  'h=$(hostname); grep -q "[[:space:]]$h\([[:space:]]\|$\)" /etc/hosts || echo "127.0.1.1 $h" >> /etc/hosts'
docker exec "$container" pkill -f "serve_policy.py.*${config}" 2>/dev/null || true
docker exec -d --user dev -e HOME=/home/dev -e PYTHONUNBUFFERED=1 "$container" bash -lc \
  "cd /home/dev/workspace/openpi_deploy/repos/openpi && exec uv run scripts/serve_policy.py --port '$port' policy:checkpoint --policy.config='$config' --policy.dir='$container_stage' > '$log' 2>&1" \
  </dev/null >/dev/null 2>&1
exit 0
REMOTE

for _ in $(seq 1 60); do
  if port_open; then
    echo "[POLICY] ready: ${SERVER}:${PORT} config=${CONFIG} checkpoint=${CHECKPOINT}"
    exit 0
  fi
  sleep 2
done

echo "[POLICY] ERROR: server did not open ${SERVER}:${PORT}" >&2
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=5 "dev@${SERVER}" "tail -n 40 '$LOG'" >&2 || true
exit 1
