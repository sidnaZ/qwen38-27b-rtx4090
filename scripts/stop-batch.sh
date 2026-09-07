#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-batch}
load_profile "$profile"
"${COMPOSE[@]}" --profile batch down
