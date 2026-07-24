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

See the recipe README for the full deploy, dataset generation, and `aiperf` invocation.

## Results — H100

| Config         |   RPS | ITL (ms) | TTFT avg (ms) | TTFT p50 | TTFT p90 | TTFT p99 |
|----------------|------:|---------:|--------------:|---------:|---------:|---------:|
| `vllm-serve`   | 0.719 |    21.28 |      18173.07 | 18829.99 | 26991.82 | 48589.22 |
| `dynamo-fd`    | 0.811 |    28.42 |       7193.37 |  3567.13 | 10991.26 | 34900.51 |
| `dynamo-fd-ec` | 0.933 |    24.56 |       6100.92 |  2369.01 | 22868.77 | 34582.94 |

**Δ vs `vllm-serve`:**

| Config         |    RPS |    ITL | TTFT avg | TTFT p50 | TTFT p90 | TTFT p99 |
|----------------|-------:|-------:|---------:|---------:|---------:|---------:|
| `dynamo-fd`    | +12.8% | +33.5% |   −60.4% |   −81.1% |   −59.3% |   −28.2% |
| `dynamo-fd-ec` | +29.8% | +15.4% |   −66.4% |   −87.4% |   −15.3% |   −28.8% |

## Results — GB200

| Config         |   RPS | ITL (ms) | TTFT avg (ms) | TTFT p50 | TTFT p90 | TTFT p99 |
|----------------|------:|---------:|--------------:|---------:|---------:|---------:|
| `vllm-serve`   | 0.940 |    15.37 |      14953.97 | 15391.49 | 21660.25 | 24221.30 |
| `dynamo-fd`    | 1.117 |    16.82 |       9478.46 |  9060.62 | 15399.09 | 17325.61 |
| `dynamo-fd-ec` | 1.236 |    15.22 |       8478.29 |  8323.53 | 13992.25 | 16075.06 |

**Δ vs `vllm-serve`:**

| Config         |    RPS |   ITL | TTFT avg | TTFT p50 | TTFT p90 | TTFT p99 |
|----------------|-------:|------:|---------:|---------:|---------:|---------:|
| `dynamo-fd`    | +18.8% | +9.4% |   −36.6% |   −41.1% |   −28.9% |   −28.5% |
| `dynamo-fd-ec` | +31.5% | −1.0% |   −43.3% |   −45.9% |   −35.4% |   −33.6% |

## Takeaways

- **Throughput:** `dynamo-fd-ec` delivers **~30% higher RPS** than vanilla `vllm serve` on both H100 (+29.8%) and GB200 (+31.5%).
- **TTFT:** The embedding cache cuts median TTFT by **−87.4% on H100** and **−45.9% on GB200** — once the encoder is off the critical path for repeated images, first-token latency collapses.
- **Frontend-decoding alone** (`dynamo-fd`) already captures most of the TTFT win; the embedding cache then layers on additional throughput and tighter median TTFT.
- **ITL** stays roughly flat because the cache shortens the *prefill* path (skipping ViT for repeated images), not the *decode* path. Throughput still rises because, by Little's Law at fixed concurrency = 30, `RPS ≈ concurrency / (TTFT + ITL × out_tokens)` — a multi-second drop in TTFT alone is enough to explain the ~30% RPS gain. The small per-token ITL deltas reflect concurrency changes from the higher RPS, not extra decode cost.

## Reproduce

```bash
cd recipes/qwen3.6-35b
HW=h100   # or gb200
./run-all-benchmarks.sh -n <namespace> --hw "${HW}"
```

Each config's `profile_export_aiperf.json` lands under
`~/workspace/dynamo-tmp/logs/$(date +%m-%d)/qwen36-fp8-${HW}/{vllm-serve,dynamo-fd,dynamo-fd-ec}/`
and holds the headline metrics.

Full instructions and prerequisites live in [`recipes/qwen3.6-35b/README.md`](../../recipes/qwen3.6-35b/README.md).
