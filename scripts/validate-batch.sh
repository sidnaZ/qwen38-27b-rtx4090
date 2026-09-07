#!/usr/bin/env bash
# Validate the active batch service and its selected profile.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-batch}
case "$profile" in batch|batch-max) ;; *) echo "usage: $0 [batch|batch-max]" >&2; exit 2 ;; esac
load_profile "$profile"
require_host
require_power 0
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

cid=$(container_id batch)
[ -n "$cid" ] || { echo "no batch container; launch $profile first" >&2; exit 1; }

expected=$(printf '%s\n' \
  "EXTRA_ARGS=$EXTRA_ARGS" "GPU_UTIL=$GPU_UTIL" "INT8_ACT=$INT8_ACT" \
  "INT8_LAYERS=$INT8_LAYERS" "KV=$KV" "MAX_LEN=$MAX_LEN" \
  "MAX_SEQS=$MAX_SEQS" "PREFIX_CACHE=$PREFIX_CACHE" "TOOLS=$TOOLS" | sort)
actual=$(docker inspect "$cid" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(EXTRA_ARGS|GPU_UTIL|INT8_ACT|INT8_LAYERS|KV|MAX_LEN|MAX_SEQS|PREFIX_CACHE|TOOLS)=' | sort)
[ "$actual" = "$expected" ] || {
  echo "container environment does not match $PROFILE_NAME" >&2
  diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
  exit 1
}

logs=$(docker logs "$cid" 2>&1)
fatal_re='CUDA error|CUDA out of memory|OutOfMemoryError|EngineDeadError|Traceback|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)'
if grep -Eiq "$fatal_re" <<<"$logs"; then
  echo "fatal CUDA/engine signature found in batch log" >&2
  grep -Ei "$fatal_re" <<<"$logs" >&2
  exit 1
fi

"${COMPOSE[@]}" exec -T batch bash verify.sh
EXPECT_THINKING_BUDGET_REJECTED=0 PORT=$PORT python3 "$PROJECT_ROOT/bench/api_smoke.py"
memory=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
echo "batch VALIDATION PASS profile=$PROFILE_NAME power=${POWER_LIMIT_W}W vram=${memory}MiB api=12/12"
