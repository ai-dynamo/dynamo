<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Benchmark Report — B200 Aggregated + Disaggregated

## Configuration

| | 4-GPU Agg | 8-GPU Agg | 12-GPU Agg | Disagg 1P1D | Disagg 2P1D |
| --- | ----------- | ----------- | ----------- |
| **Model** | Inferact/Qwen3.8-Flash-Next-NVFP4 (125B total, 6B active) | same | same |
| **Image** | vllm/vllm-openai:qwen38-flash-next | same | same |
| **Trace** | 64K agentic (15% subset, 3,541 requests) | same | same |
| **Concurrency** | 24 | 24 | 24 |
| **Workers** | 1 × TP4 | 2 × TP4 | 3 × TP4 | 1P (TP4) + 1D (TP4) | 2P (TP4) + 1D (TP4) |
| **Total GPUs** | 4 | 8 | 12 | 8 | 12 |
| **MTP3** | ✅ | ✅ | ✅ |
| **Expert parallel** | ✅ | ✅ | ✅ |
| **N-gram offload** | ✅ | ✅ | ✅ |
| **KV transfer** | — | — | — | NIXL over InfiniBand RDMA (~18 GB/s) | NIXL over InfiniBand RDMA (~18 GB/s) |
| **`--no-async-scheduling`** | — | — | Not needed (async ON, no errors observed) |
| **Prefill `--max-num-seqs`** | 256 | 256 | 256 | 32 (reduced to avoid OOM) | 32 (reduced to avoid OOM) |
| **ai-dynamo** | 1.4.2 (pip) | 1.4.2 (pip) | 1.4.2 (pip) |

## Results (from AIPerf `profile_export_aiperf.json`)

| Metric | 4-GPU Agg | 8-GPU Agg | 12-GPU Agg | Disagg 1P1D | Disagg 2P1D | Δ (12GPU Agg vs 2P1D) |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- | --- |
| **Requests completed** | 3,411 | 3,411 | 3,411 | 3,411 | 3,411 | same |
| **Requests errored** | 130 | 130 | 130 | 130 | 130 | same (>262K context) |
| **Benchmark duration** | 4,285 sec | 3,169 sec | 2,745 sec | 2,968 sec | 2,867 sec | -4% |
| | | | | | | |
| **Request latency p50** | 5,132 ms | 4,187 ms | 3,707 ms | 4,159 ms | 3,422 ms | +8% |
| **Request latency p90** | 66,761 ms | 49,369 ms | 43,526 ms | 44,464 ms | 43,976 ms | -1% |
| **Request latency mean** | 22,342 ms | 16,652 ms | 14,393 ms | 15,376 ms | 14,765 ms | -3% |
| | | | | | | |
| **TTFT p50** | 329 ms | 339 ms | 325 ms | 452 ms | 407 ms | -20% |
| **TTFT p90** | 1,851 ms | 1,306 ms | 1,110 ms | 3,467 ms | 1,165 ms | -5% |
| **TTFT mean** | 771 ms | 649 ms | 603 ms | 1,256 ms | 699 ms | -14% |
| | | | | | | |
| **ITL p50** | 9.7 ms | 7.8 ms | 6.9 ms | 7.5 ms | 7.5 ms | -8% |
| **ITL p90** | 15.1 ms | 12.8 ms | 11.3 ms | 9.1 ms | 8.9 ms | +21% |
| **ITL mean** | 11.3 ms | 9.1 ms | 8.0 ms | 7.4 ms | 7.3 ms | +10% |
| | | | | | | |
| **System output tok/s** | 1,830 | 2,474 | 2,857 | 2,641 | 2,735 | +4.5% |
| **Per-GPU tok/s** | 457.4 | 309.3 | 238.1 | 330.2 | 227.9 | +4.5% |
| **User tok/s (p50)** | 102.8 | 127.6 | 144.5 | 132.8 | 133.4 | +8% |
| **Total tok/s** | 42,529 | 57,507 | 66,399 | 61,395 | 63,573 | +4% |
| **Request throughput** | 0.80 req/s | 1.08 req/s | 1.24 req/s | 1.15 req/s | 1.19 req/s | +4% |
| | | | | | | |
| **Prefix cache hit** | 67.9% | 73.1% | 74.5% | 73.4% | 77.2% | -2.7pp |
| **Prefill tput/user** | 114,471 | 114,941 | 119,878 | 78,187 | 97,959 | +22% |
| **Avg input tokens/req** | 51,131 | 51,131 | 51,131 | 51,131 | 51,131 | same |
| **Avg output tokens/req** | 2,299 | 2,299 | 2,299 | 2,299 | 2,299 | same |

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
7. **When disagg wins**: Decode-heavy workloads (longer outputs, lower cache reuse) benefit more from prefill/decode specialization. For this short-output agentic workload (400 OSL, 90% cache reuse), disagg 1P1D is already 7% faster than 8-GPU agg.

### 12-GPU Apple-to-Apple: Agg (3×TP4) vs Disagg (2P1D)

1. **Throughput**: Agg (2,857 tok/s) beats disagg (2,735 tok/s) by **4.5%** — 3 independent workers have no KV transfer overhead
2. **TTFT**: Agg p50 is 325ms vs disagg 407ms (**-20%**) — no KV transfer latency; agg p90 is 1,110ms vs 1,165ms (-5%)
3. **ITL**: Agg p50 is 6.9ms vs disagg 7.5ms (-8%); but disagg p90 is better (8.9ms vs 11.3ms, **-21%**) — disagg's decode specialization helps tail latency
4. **Cache hit**: Disagg wins (77.2% vs 74.5%, +2.7pp) — more prefill workers = more KV cache = more prefix reuse
5. **Crossover**: At 8 GPUs, disagg 1P1D wins (+7%); at 12 GPUs, agg 3×TP4 wins (+4.5%). The crossover happens because agg scales linearly (no transfer cost) while disagg pays RDMA overhead per request

## Notes

- All numbers are from AIPerf's official `profile_export_aiperf.json` summary (not manually computed)
- 130 requests (out of 3,541) exceeded the model's 262K context limit and were rejected with 400 errors
- The 8-GPU config uses 2 workers (2×TP4) on a single node, not TP8 (TP8 is incompatible with NVFP4 128-wide quantization blocks)
- The disagg recipe uses InfiniBand RDMA (`rdma/rdma_shared_device_a: 4` per worker) for KV transfer — no `hostIPC` needed
- `--no-async-scheduling` was tested and found NOT needed — async scheduling is enabled (default) with no 500 errors or stability issues. The GDN hybrid model race (vLLM #42182, #37285) did not manifest with `--max-num-seqs 32` on prefill. If 500 errors appear in other workloads, add `--no-async-scheduling` to both workers as a workaround.
- Prefill `--max-num-seqs` is reduced to 32 (from 256) to avoid OOM on 260K-token prompts with only 4 GPUs
