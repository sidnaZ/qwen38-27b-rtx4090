#!/usr/bin/env bash
# Launch the recommended batch profile or the experimental batch-max profile.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"

profile=${1:-batch}
case "$profile" in batch|batch-max) ;; *) echo "usage: $0 [batch|batch-max] [--set-power-limit]" >&2; exit 2 ;; esac
shift || true
set_power=0
case "${1:-}" in
  --set-power-limit) set_power=1; shift ;;
  "") ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { echo "unexpected arguments: $*" >&2; exit 2; }

load_profile "$profile"
require_host
require_power "$set_power"
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "missing $IMAGE; run scripts/prepare.sh first" >&2
  exit 1
}

"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" --profile batch up -d --force-recreate batch
wait_healthy batch

cid=$(container_id batch)
echo "$PROFILE_NAME is healthy at http://127.0.0.1:${PORT}"
echo "power=${POWER_LIMIT_W}W container=$cid image=$IMAGE"
