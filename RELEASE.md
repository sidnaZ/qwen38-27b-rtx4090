# Release results

This page records the measured operating envelopes. A local build uses
`qwen38-27b-rtx4090:latest`; rerun validation and benchmarks after rebuilding
because an archived Docker image ID is evidence, not a portable image name.

## Test system

| Item | Validated value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, SM89, 24,564 MiB |
| Driver / container CUDA | 595.84 / 13.0 |
| OS | Ubuntu 24.04.4, native Linux |
| Python / vLLM | 3.12.3 / 0.27.1 |
| PyTorch / Triton | 2.13.0+cu130 / 3.7.1 |
| Locally rebuilt image | `sha256:059d52965802e0506a2b092a71255f98fe7ab439414bd85aac0b4e51de546df3` |

Exact source, model, container, and dataset pins are in
[`release-manifest.json`](release-manifest.json).

## Single-user

The `single-user` profile uses DFlash2 k=7, lookup chains, BF16 KV,
FlashAttention, prefix caching, and a 65,536-token configured context at 450 W.
The clean repository rebuild passed the full validation gate. One discarded
warmup and three cold-prefix measured repetitions produced:

- 237.9 tok/s median sustained decode.
- 337.8 tok/s median context-copy decode.
- PPL 8.0930 over 32,635 scored tokens.
- GSM8K 193/200 and API smoke 12/12.
- 23,321 MiB during final validation.

`single-user-efficient` keeps the inference configuration and lowers the power
limit to 250 W. It measured 160.5 tok/s median on the inherited eight-prompt
comparison protocol.

## Batch

The recommended `batch` profile uses W4A16 execution, FP8 KV, no speculation,
C32 admission, a fixed 5 GiB KV cache, and 350 W. Three synchronized short/512
runs from the clean repository rebuild averaged:

- 1,318.4 aggregate decode tok/s.
- 1,141.4 aggregate end-to-end output tok/s.
- 21,521 MiB peak, with no failures or preemptions.

It measured PPL 8.0482017 and GSM8K 1249/1319 versus 1252/1319 on a fresh
single-user comparison. Paired discordance was balanced: 13 single-user-only
correct and 10 batch-only correct. C32/1,024 completed without preemption, and
retrieval passed at a 99,982-token prompt.

The experimental `batch-max` profile uses activation INT8 on all linear layers,
C64 admission, and 450 W. The clean rebuild averaged 2,029.8 aggregate decode
and 1,803.3 aggregate end-to-end output tok/s. The frozen quality campaign
measured PPL 8.3806865 (+3.566%), so this is not the quality-first default.
C64/1,024 preempted; C48/1,024 completed cleanly.

Synchronized cohorts measure resident capacity rather than realistic arrival
traffic. Aggregate batch throughput must not be compared to single-request
decode speed.
