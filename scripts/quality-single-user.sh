#!/usr/bin/env bash
# Re-run the frozen PPL + GSM8K quality battery inside the serving container.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-single-user}
load_profile "$profile"
require_host
require_power 0
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
tag=${2:-${profile}-$(date -u +%Y%m%dT%H%M%SZ)}
"${COMPOSE[@]}" exec -T single-user /app/venv/bin/python bench/quality_battery.py "$tag"
