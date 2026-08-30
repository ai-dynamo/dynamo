<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Benchmark Report — B200 Aggregated

## Configuration

| | 4-GPU Agg | 8-GPU Agg |
| --- | ----------- | ----------- |
| **Model** | Inferact/Qwen3.8-Flash-Next-NVFP4 (125B total, 6B active) | same |
| **Image** | vllm/vllm-openai:qwen38-flash-next | same |
| **Trace** | 64K agentic (15% subset, 3,541 requests) | same |
| **Concurrency** | 24 | 24 |
| **Workers** | 1 × TP4 | 2 × TP4 |
| **Total GPUs** | 4 | 8 |
| **MTP3** | ✅ | ✅ |
| **Expert parallel** | ✅ | ✅ |
| **N-gram offload** | ✅ | ✅ |
| **ai-dynamo** | 1.4.2 (pip) | 1.4.2 (pip) |

## Results (from AIPerf `profile_export_aiperf.json`)

| Metric | 4-GPU Agg | 8-GPU Agg | Δ |
| -------- | ----------- | ----------- | --- |
| **Requests completed** | 3,411 | 3,411 | same |
| **Requests errored** | 130 | 130 | same (>262K context) |
| **Benchmark duration** | 4,285 sec | 3,169 sec | -26% |
| | | | |
| **Request latency p50** | 5,132 ms | 4,187 ms | -18% |
| **Request latency p90** | 66,761 ms | 49,369 ms | -26% |
| **Request latency mean** | 22,342 ms | 16,652 ms | -25% |
| | | | |
| **TTFT p50** | 329 ms | 339 ms | +3% |
| **TTFT p90** | 1,851 ms | 1,306 ms | -29% |
| **TTFT mean** | 771 ms | 649 ms | -16% |
| | | | |
| **ITL p50** | 9.7 ms | 7.8 ms | -20% |
| **ITL p90** | 15.1 ms | 12.8 ms | -15% |
| **ITL mean** | 11.3 ms | 9.1 ms | -19% |
| | | | |
| **System output tok/s** | 1,830 | 2,474 | +35% |
| **Per-GPU tok/s** | 457.4 | 309.3 | -32% |
| **User tok/s (p50)** | 102.8 | 127.6 | +24% |
| **Total tok/s** | 42,529 | 57,507 | +35% |
| **Request throughput** | 0.80 req/s | 1.08 req/s | +35% |
| | | | |
| **Prefix cache hit** | 67.9% | 73.1% | +5.2pp |
| **Prefill tput/user** | 114,471 | 114,941 | same |
| **Avg input tokens/req** | 51,131 | 51,131 | same |
| **Avg output tokens/req** | 2,299 | 2,299 | same |

## Key Findings

1. **Aggregate throughput**: 8-GPU is **1.4× faster** (2,474 vs 1,830 tok/s) — doubling GPUs gives +35% aggregate
2. **Per-GPU efficiency**: 4-GPU is **1.5× more efficient** per GPU (458 vs 309 tok/s/GPU) — the model is so sparse (6B active) that 4 GPUs already saturate compute
3. **TTFT**: Nearly identical at p50 (329 vs 339ms) — prefill is compute-bound, not memory-bound
4. **ITL**: 8-GPU is **20% faster** (9.7 → 7.8 ms) — more KV cache headroom per request
5. **Prefix cache hit**: 8-GPU has **+5pp higher** hit rate (73.1% vs 67.9%) — more total KV cache = more prefix reuse
6. **User experience**: Similar at p50 (103 vs 128 tok/s/user) — both deliver excellent interactivity
7. **p90 latency**: 8-GPU significantly better (49K vs 67K ms) — more GPUs absorb the long-tail requests
8. **130 errors** in both runs — requests exceeding 262K context limit (expected, trace has some 1M+ token requests)

## Notes

- All numbers are from AIPerf's official `profile_export_aiperf.json` summary (not manually computed)
- 130 requests (out of 3,541) exceeded the model's 262K context limit and were rejected with 400 errors
- The 8-GPU config uses 2 workers (2×TP4) on a single node, not TP8 (TP8 is incompatible with NVFP4 128-wide quantization blocks)
- Disaggregated benchmark was attempted but encountered stability issues with `hostIPC` + `cuda_ipc` — will be revisited in a follow-up
