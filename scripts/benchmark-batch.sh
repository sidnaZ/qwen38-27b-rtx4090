#!/usr/bin/env bash
# Forward a batch benchmark command after checking the selected live profile.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-batch}
shift || true
load_profile "$profile"
require_host
require_power 0
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
python3 "$PROJECT_ROOT/bench/benchmark-batch.py" "$@"
