# Core vLLM patches

Files in this directory are part of the RTX 4090 serving stack. The Docker
build applies every `*.patch` here automatically.

## GDN speculative-decoding bounds check

`vllm-pr50021-gdn-spec-bounds.patch` is a core runtime-correctness patch, not a
Codex integration workaround. It bounds accepted-token state lookups in the
Gated DeltaNet/Mamba speculative-decoding kernels. Without it, stale or
oversized accepted-token counts can address memory outside the recurrent-state
row and cause illegal GPU-memory accesses under concurrent MTP requests.

The two Responses API tool-call patches are deliberately not stored here.
They live in [`integrations/codex/`](../integrations/codex/README.md) because
they address tool-using Responses clients such as local Codex rather than GPU
execution, quantization, or RTX 4090 performance.
