#!/usr/bin/env bash
# Fixed low-latency profile for one active user on an RTX 4090.

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

MODEL=${MODEL:-}
if [ -z "$MODEL" ] && [ -d models/Qwen3.8-27B-W4A16-AutoRound-fast ]; then
  MODEL=$PROJECT_ROOT/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$PROJECT_ROOT/models/Qwen3.8-27B-W4A16-AutoRound}
DRAFT=${DRAFT:-$PROJECT_ROOT/models/Qwen3.8-27B-DFlash2-W4A16}
[ -f "$DRAFT/model.safetensors" ] || {
  echo "DFlash2 drafter missing at $DRAFT; run scripts/prepare.sh" >&2
  exit 1
}

PORT=${PORT:-18020}
GPU_UTIL=${GPU_UTIL:-0.93}
MAX_LEN=${MAX_LEN:-65536}
MAX_SEQS=${MAX_SEQS:-8}
DRAFT_TOKENS=${DFLASH_TOKENS:-7}
KV_MEM=${KV_MEM:-5583457484}
API_SERVERS=${API_SERVERS:-1}
CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}

export PATH="$PROJECT_ROOT/venv/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}

if [ -z "${VLLM_API_KEY:-}" ] && [ -f api_key.txt ]; then
  export VLLM_API_KEY
  VLLM_API_KEY=$(cat api_key.txt)
fi

thinking_args=()
if [ "${THINKING:-1}" = 0 ]; then
  thinking_args=(--default-chat-template-kwargs '{"enable_thinking":false}')
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-memory "$KV_MEM" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --api-server-count "$API_SERVERS" \
  --language-model-only \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "${thinking_args[@]}" \
  ${EXTRA_ARGS:-}
