#!/usr/bin/env bash
# Fixed high-throughput profile for concurrent requests on an RTX 4090.

set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for region in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$region" ] || continue
    grep -lqs "$region" /proc/[0-9]*/maps 2>/dev/null || {
      echo "removing stale offload region $region"
      rm -f "$region"
    }
  done
fi

MODEL=${MODEL:-$PROJECT_ROOT/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
GPU_UTIL=${GPU_UTIL:-0.93}
MAX_LEN=${MAX_LEN:-150000}
MAX_SEQS=${MAX_SEQS:-32}
API_SERVERS=${API_SERVERS:-1}

[ "${KV:-fp8}" = fp8 ] || {
  echo "the production batch profiles support KV=fp8 only" >&2
  exit 2
}

export PATH="$PROJECT_ROOT/venv/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -n "${INT8_ACT:-}" ]; then
  export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
  export VLLM_MARLIN_INT8_INCLUDE_RE=${INT8_LAYERS:-.}
fi
if [ -z "${VLLM_API_KEY:-}" ] && [ -f api_key.txt ]; then
  export VLLM_API_KEY
  VLLM_API_KEY=$(cat api_key.txt)
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --api-server-count "$API_SERVERS" \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  ${EXTRA_ARGS:-}
