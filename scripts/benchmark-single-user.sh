#!/usr/bin/env bash
# Run the frozen six-workload coding suite against an already healthy single-user service.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-single-user}
load_profile "$profile"
require_host
require_power 0
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
cid=$(container_id single-user)
[ -n "$cid" ] || { echo "no single-user container" >&2; exit 1; }
image_id=$(docker inspect "$cid" --format '{{.Image}}')
tag=${2:-${profile}-$(date -u +%Y%m%dT%H%M%SZ)}
python3 "$PROJECT_ROOT/benchmarks/run_coding_suite.py" \
  --tag "$tag" \
  --output-dir "$BENCH_RESULTS_DIR/coding-suite" \
  --image-id "$image_id" \
  --config PROFILE_NAME="$profile" \
  --config SPEC=dflash2 --config CTX=fast --config PREFIX_CACHE=1 \
  --config DFLASH_TOKENS=7 --config SPEC_ATTN=1 --config VLLM_DFLASH2_CHAIN=1
