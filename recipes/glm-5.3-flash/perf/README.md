<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash — Benchmark Guide

This guide uses [AIPerf](https://github.com/ai-dynamo/aiperf) to benchmark a deployed
GLM-5.3-Flash endpoint against the agentic workload profile (64K ISL, 400 OSL, 90% KV
cache reuse).

## Prerequisites

- A running GLM-5.3-Flash deployment (see `../README.md`)
- `aiperf` installed: `pip install aiperf`
- The frontend service endpoint (`ENDPOINT`)

## Benchmark

### 1. Set environment

```bash
export ENDPOINT="http://<frontend-service>:8000"
export MODEL="zai-org/GLM-5.3-Flash"
```

### 2. Warm up

Run a single-request warm-up to prime KV cache and CUDA graphs before the sweep:

```bash
aiperf benchmark \
  --base-url "${ENDPOINT}/v1" \
  --model "${MODEL}" \
  --workload agentic \
  --isl 6400 \
  --osl 400 \
  --num-requests 1 \
  --warmup 1 \
  --streaming \
  --use-server-token-count
```

### 3. Concurrency sweep

Sweep concurrency from a point near saturation. For 4× GB200 TP4 agg, start at C=4:

```bash
for C in 4 8 16 32 48 64; do
  aiperf benchmark \
    --base-url "${ENDPOINT}/v1" \
    --model "${MODEL}" \
    --workload agentic \
    --isl 6400 \
    --osl 400 \
    --concurrency "${C}" \
    --num-requests 200 \
    --warmup 1 \
    --streaming \
    --use-server-token-count \
    --cache-salt "$(date +%s%N)" \
    2>&1 | tee "aiperf_C${C}.log"
done
```

> **Note**: Pass `--cache-salt` on each run to prevent cross-run prefix-cache pollution.

### 4. Report

Key metrics to record per concurrency:

| Metric | aiperf field |
| ------ | ------------ |
| System output tok/s | `output_tokens_per_second` |
| User output tok/s (P50) | `itl_p50` × tokens |
| TTFT P50 (ms) | `ttft_p50_ms` |
| E2E latency P99 (ms) | `e2e_latency_p99_ms` |

The target operating point is the concurrency where system throughput peaks while TTFT
P50 remains below 5000 ms for the 64K ISL agentic workload.

## Reference Results

### GB200 Aggregated (`agg-gb200-agentic`) — TP4, 4 GPUs

Hardware: GB200 NVL72 | `vllm/vllm-openai:glm53-flash` + `ai-dynamo==1.4.1`  
Workload: SSP=57600, ISL=6400, OSL=400, 90% KV reuse

| Concurrency | tok/s/user | tok/s (total) | tok/s/GPU | ITL p50 (ms) | TTFT avg (ms) | SLA ≥50 |
|-------------|-----------|---------------|-----------|--------------|---------------|---------|
| 4  | 95.95 | 271.64  | 67.9  | 10.39 | 1,662 | ✓ |
| 8  | 67.54 | 379.75  | 94.9  | 15.45 | 2,338 | ✓ |
| 16 | 42.19 | 501.74  | 125.4 | 27.22 | 2,863 | ✗ |
| 32 | 27.23 | 702.81  | 175.7 | 39.36 | 2,988 | ✗ |

SLA (50 tok/s/user) crosses between C=8 and C=16. Interpolated tok/s/GPU at crossing ≈ **108**.
