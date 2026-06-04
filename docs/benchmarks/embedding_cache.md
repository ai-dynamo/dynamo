---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Embedding Cache Benchmark
subtitle: Throughput and TTFT gains from caching vision-encoder embeddings on a multimodal workload
---

## Overview

This page reports the performance impact of Dynamo's multimodal frontend-decoding and embedding cache on a repeated-image workload, measured against a vanilla `vllm serve` baseline.

For the feature itself — what the embedding cache is, support across backends (vLLM / TRT-LLM / SGLang), aggregated vs disaggregated workflows, and how to enable it — see [Embedding Cache](../features/multimodal/embedding-cache.md).

## Recipe

The numbers below come from the [`qwen3.6-35b` recipe](../../recipes/qwen3.6-35b/README.md), which runs three configurations on the same single-GPU node so the only thing that varies is the Dynamo feature set:

| Config         | Stack          | Frontend-decoding | Embedding cache |
|----------------|----------------|-------------------|-----------------|
| `vllm-serve`   | vanilla vLLM   | n/a               | n/a             |
| `dynamo-fd`    | Dynamo + vLLM  | on                | off             |
| `dynamo-fd-ec` | Dynamo + vLLM  | on                | 8 GiB           |

**Workload:** `Qwen/Qwen3.6-35B-A3B-FP8`, sliding-window dataset (30 users × 8 turns, window = 5, 2400×1080 base64 images, 8000 input text tokens, `max_tokens=1024`, concurrency = 30). Each turn shares 4-of-5 images with the previous turn of the same user, so repeated images dominate — the exact shape the embedding cache is designed for.

See the [recipe README](../../recipes/qwen3.6-35b/README.md) for standalone deploy instructions, and the [benchmark README](../../recipes/qwen3.6-35b/benchmark/README.md) for dataset generation and the `aiperf` invocation.

## Results — H100

Change vs the `vllm-serve` baseline (negative TTFT/ITL = lower latency, better):

| Config         |    RPS |    ITL | TTFT avg | TTFT p50 | TTFT p90 | TTFT p99 |
|----------------|-------:|-------:|---------:|---------:|---------:|---------:|
| `dynamo-fd`    | +12.8% | +33.5% |   −60.4% |   −81.1% |   −59.3% |   −28.2% |
| `dynamo-fd-ec` | +29.8% | +15.4% |   −66.4% |   −87.4% |   −15.3% |   −28.8% |

## Takeaways

- **Throughput:** `dynamo-fd-ec` delivers **~30% higher RPS** (+29.8%) than vanilla `vllm serve`.
- **TTFT:** The embedding cache cuts median TTFT by **−87.4%** — once the encoder is off the critical path for repeated images, first-token latency collapses.
- **Frontend-decoding alone** (`dynamo-fd`) already captures most of the TTFT win; the embedding cache then layers on additional throughput and tighter median TTFT.
- **ITL** stays roughly flat because the cache shortens the *prefill* path (skipping ViT for repeated images), not the *decode* path. Throughput still rises because, by Little's Law at fixed concurrency = 30, `RPS ≈ concurrency / (TTFT + ITL × out_tokens)` — a large drop in TTFT alone is enough to explain the ~30% RPS gain. The small per-token ITL deltas reflect concurrency changes from the higher RPS, not extra decode cost.

## Reproduce

```bash
cd recipes/qwen3.6-35b/benchmark
HW=h100   # or gb200
./run-all-benchmarks.sh -n <namespace> --hw "${HW}"
```

Each config's `profile_export_aiperf.json` lands under
`~/workspace/dynamo-tmp/logs/$(date +%m-%d)/qwen36-fp8-${HW}/{vllm-serve,dynamo-fd,dynamo-fd-ec}/`
and holds the headline metrics.

Full instructions and prerequisites live in [`recipes/qwen3.6-35b/README.md`](../../recipes/qwen3.6-35b/README.md).
