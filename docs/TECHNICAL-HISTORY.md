# Technical history

The project started from
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090),
then reproduced, profiled, and retuned that stack for RTX 4090 / SM89.

The selected single-user path combines a W4A16 target, quantized heads, a W4A16
DFlash2 drafter, lookup chains, split-KV verification, BF16 KV, FlashAttention,
and prefix caching. Final release validation measured 237.9 tok/s sustained
decode, 337.8 tok/s context-copy, PPL 8.0930, GSM8K 193/200, and API 12/12.

Profiling showed that target-weight traffic dominates concurrency-one decode.
SM89 Marlin experiments, adaptive speculative depth, and recurrent/GDN fusion
improved local kernels but did not produce a material whole-system gain under
the release gates. Fixed DFlash2 k=7 remained the robust default.

At high concurrency, speculative state and verify work stop paying for
themselves. The quality-first batch profile therefore uses W4A16 without
speculation, FP8 KV, C32 admission, and 350 W. It measured 1,318.4 aggregate
decode and 1,141.4 aggregate end-to-end output tok/s, with quality comparable to
single-user in paired/full-split evaluation.

All-linear activation INT8 reaches 2,029.8 aggregate decode and 1,803.3
aggregate end-to-end output tok/s at C64, but increased perplexity by 3.566%.
That configuration remains the explicitly experimental `batch-max` profile.

Async scheduling improved C64 decode by about 4%. A larger 4,096-token scheduler
budget added essentially no decode throughput and consumed another 618 MiB, so
it was rejected. C64/1,024 preempted; C48/1,024 completed cleanly.

The development campaign also produced phase reports, profiler overlays,
generated demos, alternative-checkpoint tools, and one-off microbenchmarks.
They are intentionally absent from the production repository. The retained
files are either on a supported build/run/validation path, pin its provenance,
or provide the exact benchmark inputs needed to reproduce published results.
