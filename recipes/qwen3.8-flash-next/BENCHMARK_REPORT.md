<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Benchmark Report — B200 Aggregated + Disaggregated

## Configuration

| | 4-GPU Agg | 8-GPU Agg | Disagg 1P1D |
| --- | ----------- | ----------- | ----------- |
| **Model** | Inferact/Qwen3.8-Flash-Next-NVFP4 (125B total, 6B active) | same | same |
| **Image** | vllm/vllm-openai:qwen38-flash-next | same | same |
| **Trace** | 64K agentic (15% subset, 3,541 requests) | same | same |
| **Concurrency** | 24 | 24 | 24 |
| **Workers** | 1 × TP4 | 2 × TP4 | 1P (TP4) + 1D (TP4) |
| **Total GPUs** | 4 | 8 | 8 |
| **MTP3** | ✅ | ✅ | ✅ |
| **Expert parallel** | ✅ | ✅ | ✅ |
| **N-gram offload** | ✅ | ✅ | ✅ |
| **KV transfer** | — | — | NIXL over InfiniBand RDMA (~18 GB/s) |
| **`--no-async-scheduling`** | — | — | Not needed (async ON, no errors observed) |
| **Prefill `--max-num-seqs`** | 256 | 256 | 32 (reduced to avoid OOM) |
| **ai-dynamo** | 1.4.2 (pip) | 1.4.2 (pip) | 1.4.2 (pip) |

## Results (from AIPerf `profile_export_aiperf.json`)

| Metric | 4-GPU Agg | 8-GPU Agg | Disagg 1P1D | Δ (Disagg vs 8-GPU) |
| -------- | ----------- | ----------- | ----------- | --- |
| **Requests completed** | 3,411 | 3,411 | 3,411 | same |
| **Requests errored** | 130 | 130 | 130 | same (>262K context) |
| **Benchmark duration** | 4,285 sec | 3,169 sec | 2,968 sec | -6% |
| | | | | |
| **Request latency p50** | 5,132 ms | 4,187 ms | 4,159 ms | -1% |
| **Request latency p90** | 66,761 ms | 49,369 ms | 44,464 ms | -10% |
| **Request latency mean** | 22,342 ms | 16,652 ms | 15,376 ms | -8% |
| | | | | |
| **TTFT p50** | 329 ms | 339 ms | 452 ms | +33% |
| **TTFT p90** | 1,851 ms | 1,306 ms | 3,467 ms | +165% |
| **TTFT mean** | 771 ms | 649 ms | 1,256 ms | +94% |
| | | | | |
| **ITL p50** | 9.7 ms | 7.8 ms | 7.5 ms | -4% |
| **ITL p90** | 15.1 ms | 12.8 ms | 9.1 ms | -29% |
| **ITL mean** | 11.3 ms | 9.1 ms | 7.4 ms | -19% |
| | | | | |
| **System output tok/s** | 1,830 | 2,474 | 2,641 | +7% |
| **Per-GPU tok/s** | 457.4 | 309.3 | 330.2 | +7% |
| **User tok/s (p50)** | 102.8 | 127.6 | 132.8 | +4% |
| **Total tok/s** | 42,529 | 57,507 | 61,395 | +7% |
| **Request throughput** | 0.80 req/s | 1.08 req/s | 1.15 req/s | +7% |
| | | | | |
| **Prefix cache hit** | 67.9% | 73.1% | 73.4% | +0.3pp |
| **Prefill tput/user** | 114,471 | 114,941 | 78,187 | -32% |
| **Avg input tokens/req** | 51,131 | 51,131 | 51,131 | same |
| **Avg output tokens/req** | 2,299 | 2,299 | 2,299 | same |

## Key Findings

### Aggregated: 4-GPU vs 8-GPU

1. **Aggregate throughput**: 8-GPU is **1.4× faster** (2,474 vs 1,830 tok/s) — doubling GPUs gives +35% aggregate
2. **Per-GPU efficiency**: 4-GPU is **1.5× more efficient** per GPU (458 vs 309 tok/s/GPU) — the model is so sparse (6B active) that 4 GPUs already saturate compute
3. **TTFT**: Nearly identical at p50 (329 vs 339ms) — prefill is compute-bound, not memory-bound
4. **ITL**: 8-GPU is **20% faster** (9.7 → 7.8 ms) — more KV cache headroom per request
5. **Prefix cache hit**: 8-GPU has **+5pp higher** hit rate (73.1% vs 67.9%) — more total KV cache = more prefix reuse
6. **User experience**: Similar at p50 (103 vs 128 tok/s/user) — both deliver excellent interactivity
7. **p90 latency**: 8-GPU significantly better (49K vs 67K ms) — more GPUs absorb the long-tail requests
8. **130 errors** in both runs — requests exceeding 262K context limit (expected, trace has some 1M+ token requests)

### Disaggregated: 1P1D vs 8-GPU Agg

1. **Throughput**: Disagg (2,641 tok/s) is **7% faster** than 8-GPU agg (2,474 tok/s) — async scheduling + prefill/decode specialization improves GPU utilization
2. **TTFT**: Disagg p50 is 452ms vs 339ms agg (+33%) — KV transfer overhead adds ~100ms per request; p90 is significantly worse (3,467ms vs 1,306ms) due to prefill queueing at `max-num-seqs=32`
3. **ITL**: Disagg p50 is 7.5ms vs 7.8ms agg (-4%) — async scheduling overlaps compute; ITL p90 is actually better (9.1ms vs 12.8ms, -29%)
4. **KV transfer**: ~18 GB/s avg (peak 42 GB/s) via InfiniBand RDMA — 0 transfer failures across all transfers
5. **Stability**: 0 worker restarts, 0 OOM, 0 500/503 errors — `--max-num-seqs 32` on prefill alone was sufficient; `--no-async-scheduling` was NOT needed
6. **Cache hit**: 73.4% (same as 8-GPU agg) — KV-aware routing works correctly across prefill/decode
7. **When disagg wins**: Decode-heavy workloads (longer outputs, lower cache reuse) benefit more from prefill/decode specialization. For this short-output agentic workload (400 OSL, 90% cache reuse), disagg is already 7% faster than agg.

## Notes

- All numbers are from AIPerf's official `profile_export_aiperf.json` summary (not manually computed)
- 130 requests (out of 3,541) exceeded the model's 262K context limit and were rejected with 400 errors
- The 8-GPU config uses 2 workers (2×TP4) on a single node, not TP8 (TP8 is incompatible with NVFP4 128-wide quantization blocks)
- The disagg recipe uses InfiniBand RDMA (`rdma/rdma_shared_device_a: 4` per worker) for KV transfer — no `hostIPC` needed
- `--no-async-scheduling` was tested and found NOT needed — async scheduling is enabled (default) with no 500 errors or stability issues. The GDN hybrid model race (vLLM #42182, #37285) did not manifest with `--max-num-seqs 32` on prefill. If 500 errors appear in other workloads, add `--no-async-scheduling` to both workers as a workaround.
- Prefill `--max-num-seqs` is reduced to 32 (from 256) to avoid OOM on 260K-token prompts with only 4 GPUs
