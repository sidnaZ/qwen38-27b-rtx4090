#!/usr/bin/env bash
# Shared helpers for the supported single-user and batch profiles.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

load_profile() {
  local profile=$1
  case "$profile" in
    single-user|single-user-efficient|batch|batch-max) ;;
    *) echo "unknown profile: $profile" >&2; return 2 ;;
  esac

  local profile_file=${PROFILE_FILE:-"$PROJECT_ROOT/profiles/$profile.env"}
  case "$profile_file" in
    /*) ;;
    *) profile_file="$PROJECT_ROOT/$profile_file" ;;
  esac
  [ -f "$profile_file" ] || {
    echo "profile file does not exist: $profile_file" >&2
    return 2
  }

  PROFILE_FILE=$profile_file
  set -a
  . "$PROFILE_FILE"
  set +a
  [ "${PROFILE_NAME:-}" = "$profile" ] || {
    echo "profile file $PROFILE_FILE declares PROFILE_NAME=${PROFILE_NAME:-<unset>}, expected $profile" >&2
    return 2
  }

  COMPOSE=(docker compose --project-name qwen38-27b-rtx4090
    --env-file "$PROFILE_FILE"
    -f "$PROJECT_ROOT/compose.yml")
}

require_host() {
  command -v docker >/dev/null || { echo "docker is required" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; return 1; }
  local gpu
  gpu=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name,compute_cap,memory.total --format=csv,noheader)
  case "$gpu" in
    *"RTX 4090"*", 8.9,"*) ;;
    *) echo "profile requires an RTX 4090 (SM89); GPU $GPU_INDEX reports: $gpu" >&2; return 1 ;;
  esac
}

require_power() {
  local set_limit=${1:-0}
  local current
  current=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=power.limit --format=csv,noheader,nounits)
  current=${current%.*}
  if [ "$current" != "$POWER_LIMIT_W" ] && [ "$set_limit" = 1 ]; then
    echo "Setting GPU $GPU_INDEX power limit to $POWER_LIMIT_W W (sudo is required)."
    sudo nvidia-smi -i "$GPU_INDEX" --power-limit "$POWER_LIMIT_W"
    current=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=power.limit --format=csv,noheader,nounits)
    current=${current%.*}
  fi
  [ "$current" = "$POWER_LIMIT_W" ] || {
    echo "$PROFILE_NAME requires $POWER_LIMIT_W W; GPU $GPU_INDEX is at $current W." >&2
    echo "Re-run the launcher with --set-power-limit (requires sudo)." >&2
    return 2
  }
}

container_id() {
  local service=$1
  "${COMPOSE[@]}" --profile "$service" ps -q "$service"
}

wait_healthy() {
  local service=$1
  local deadline=$((SECONDS + 900))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    local cid
    cid=$(container_id "$service")
    if [ -n "$cid" ] && [ "$(docker inspect -f '{{.State.Status}}' "$cid")" = exited ]; then
      docker logs --tail 120 "$cid" >&2
      return 1
    fi
    sleep 5
  done
  echo "service did not become healthy within 900 seconds" >&2
  return 1
}
