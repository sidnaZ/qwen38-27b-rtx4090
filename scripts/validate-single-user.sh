#!/usr/bin/env bash
# Operational single-user gate: install/model verification, active-path assertions, inference,
# full API smoke, obvious numerical/CUDA failure scan, and current VRAM fit.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
profile=${1:-single-user}
load_profile "$profile"
require_host
require_power 0
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

cid=$(container_id single-user)
[ -n "$cid" ] || { echo "no single-user container; launch $profile first" >&2; exit 1; }

expected_env='CTX=fast
DFLASH_TOKENS=7
PREFIX_CACHE=1
SPEC=dflash2
SPEC_ATTN=1
VLLM_DFLASH2_CHAIN=1'
actual_env=$(docker inspect "$cid" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(CTX|DFLASH_TOKENS|PREFIX_CACHE|SPEC|SPEC_ATTN|VLLM_DFLASH2_CHAIN)=' | sort)
[ "$actual_env" = "$expected_env" ] || {
  echo "container inference environment is not the frozen single-user profile" >&2
  diff -u <(printf '%s\n' "$expected_env") <(printf '%s\n' "$actual_env") >&2 || true
  exit 1
}

logs=$(docker logs "$cid" 2>&1)
for marker in \
  'Resolved architecture: DFlash2DraftModel' \
  'DFlash2 chains on (minmatch=8)' \
  'DFlash2 lookup-augmented drafting on (k=7' \
  'Using MarlinLinearKernel' \
  'Using FLASH_ATTN attention backend' \
  'GPU KV cache size:' \
  'Graph capturing finished'; do
  grep -Fq "$marker" <<<"$logs" || { echo "missing active-path marker: $marker" >&2; exit 1; }
done
fatal_re='CUDA error|CUDA out of memory|OutOfMemoryError|EngineDeadError|Traceback|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)'
if grep -Eiq "$fatal_re" <<<"$logs"; then
  echo "fatal CUDA/engine signature found in service log" >&2
  grep -Ei "$fatal_re" <<<"$logs" >&2
  exit 1
fi

"${COMPOSE[@]}" exec -T single-user bash verify.sh
PORT=$PORT python3 "$PROJECT_ROOT/bench/api_smoke.py"

memory=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
used=${memory%%,*}; used=${used// /}
total=${memory##*,}; total=${total// /}
[ "$used" -lt "$total" ] || { echo "GPU memory is exhausted: ${used}/${total} MiB" >&2; exit 1; }
echo "single-user VALIDATION PASS profile=$PROFILE_NAME power=${POWER_LIMIT_W}W vram=${used}/${total}MiB api=12/12"
