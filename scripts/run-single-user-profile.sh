#!/usr/bin/env bash
# Launch one frozen single-user profile. Prefer the named wrappers next to this file.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"

profile=${1:-}
[ -n "$profile" ] || { echo "usage: $0 single-user|single-user-efficient [--set-power-limit]" >&2; exit 2; }
shift
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

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "missing $IMAGE; run scripts/prepare.sh first" >&2
  exit 1
fi

"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" --profile single-user up -d --force-recreate single-user
wait_healthy single-user

cid=$(container_id single-user)
echo "$PROFILE_NAME is healthy at http://127.0.0.1:${PORT}"
echo "power=${POWER_LIMIT_W}W container=$cid image=$IMAGE"
echo "config: SPEC=dflash2 CTX=fast PREFIX_CACHE=1 DFLASH_TOKENS=7 SPEC_ATTN=1 VLLM_DFLASH2_CHAIN=1"
