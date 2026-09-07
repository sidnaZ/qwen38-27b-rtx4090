# Codex / Responses API compatibility

This directory is an integration layer, not part of the RTX 4090 model port,
quantization work, or GPU performance tuning. The patches are bundled into the
image because this repository's local Qwen deployment is used by Codex, but
ordinary chat/completions workloads do not depend on them.

The numbered filenames define their required application order:

1. `10-responses-tool-tail-order.patch` fixes a vLLM Responses streaming state
   transition. With speculative decoding, one delta can contain the final
   arguments of an open tool call plus following reasoning/content. The patch
   emits the argument tail first, preventing vLLM from reopening it as a
   nameless tool call. This is a protocol-level vLLM fix and can benefit any
   tool-using Responses client, although Codex exposed the failure here.

2. `20-responses-single-tool-call.patch` enforces a Responses request with
   `parallel_tool_calls=false`. If Qwen nevertheless emits extra calls, vLLM
   retains the first named call and its continuations and drops later or
   orphaned fragments. This is primarily a defensive local-Codex compatibility
   measure; it complements the local launcher catalog setting
   `supports_parallel_tool_calls: false`.

`test_responses_tools.py` covers both behaviors. The Dockerfile applies this
directory explicitly after the core patches, and `verify.sh` checks it in a
separate section so these files cannot be mistaken for core GPU patches.

The third recently added patch,
[`patches/vllm-pr50021-gdn-spec-bounds.patch`](../../patches/vllm-pr50021-gdn-spec-bounds.patch),
remains in the core patch directory because it prevents an actual GPU kernel
out-of-bounds access and is useful independently of Codex.
