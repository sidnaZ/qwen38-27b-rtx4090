#!/usr/bin/env bash
# Stop the local-only single-user service. Optionally restore the normal 450 W limit.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != --restore-450w ]; }; then
  echo "usage: $0 [--restore-450w]" >&2
  exit 2
fi
load_profile single-user
"${COMPOSE[@]}" --profile single-user down
if [ "${1:-}" = --restore-450w ]; then
  require_host
  require_power 1
fi
