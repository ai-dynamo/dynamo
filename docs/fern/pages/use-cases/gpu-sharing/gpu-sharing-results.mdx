---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Interpreting GPU Sharing Results
subtitle: How to read the dedicated vs. shared GPU benchmark for many-model serving
---

This page explains how to read the benchmark that compares serving one model per GPU against packing two models onto each GPU. It covers what each column measures and why the percentage columns are the ones to base a decision on.

The benchmark scripts, manifests, and results on this page are co-authored by [@marckarp](https://github.com/marckarp) and [@scheckerNV](https://github.com/scheckerNV). To reproduce them, see [Reproducing the Results](#reproducing-the-results).

<Note>
Any performance results on this page are purely illustrative and are not indicative of optimal performance. Your deployment or configuration may vary.
</Note>

## Setup

We ran the same models on an 8× A100 node with Qwen3-4B, with a fixed input length of 2048 tokens, and a fixed output length of 256 tokens.

The results are pessimistic: they assume no KV caching between requests, so the entire 2048-token input sequence length (ISL) is computed on every request.

| Configuration | Models | Placement | Isolation |
|---|---|---|---|
| Baseline | 8 | One model per GPU | Dedicated GPU (baseline) |
| HAMi | 16 | Two models per GPU | Memory limits only; the CUDA driver time-slices the two models |
| GPU fractions | 16 | Two models per GPU | Memory and compute (SM) limits; the two models run concurrently |

In the HAMi and GPU fractions configurations, each model receives half of the GPU memory. With GPU fractions, each model also receives half of the GPU's streaming multiprocessors (SMs).

- **Time-slicing:** both models share all of the GPU's SMs, and the CUDA driver context switches between them, so one model's work waits while the other's runs.
- **GPU fractioning:** each model is assigned its own fixed share of the GPU's SMs and memory, so both models run concurrently without waiting on each other.

## Results

### Throughput

Concurrency, `c`, is defined as the number of concurrent requests each model has in flight. A shared GPU hosts two models, so HAMi and GPU fractions at `c=1` place the same two requests on each GPU as vanilla at `c=2`. The first column, **Requests resident/GPU**, is that shared quantity. Each row compares the three configurations under the same pressure on the hardware.

| Requests resident/GPU | Baseline concurrency | HAMi and Fractions concurrency | tok/s per GPU Baseline | tok/s per GPU HAMi | tok/s per GPU Fractions | HAMi % of Baseline | Fractions % of Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 227.5 | 101.1 | 144.4 | 44.5% | 63.5% |
| 4 | 4 | 2 | 429.1 | 193.0 | 259.2 | 45.0% | 60.4% |
| 8 | 8 | 4 | 671.2 | 343.9 | 456.1 | 51.2% | 68.0% |
| 16 | 16 | 8 | 929.2 | 572.0 | 722.8 | 61.6% | 77.8% |
| 32 | 32 | 16 | 1,141.8 | 831.4 | 1,003.4 | 72.8% | 87.9% |

Note that we expect degradation in the model performance since we are sharing the compute of the gpu between two distinct models. We highlight that using the fractional gpu method, partitioning the SMs and HBM memory, we get better performance packing multiple models onto one gpu than when using the previous state of the art, KAI+HAMi, only partitioning the memory and time-slicing the compute.

<details>
<summary>TTFT p50 (ms)</summary>

| Requests resident/GPU | Baseline concurrency | HAMi and Fractions concurrency | TTFT p50 Baseline (ms) | TTFT p50 HAMi (ms) | TTFT p50 Fractions (ms) | HAMi % of Baseline | Fractions % of Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 48.6 | 216 | 185 | 444.4% | 380.7% |
| 4 | 4 | 2 | 49.0 | 374 | 347 | 763.3% | 708.2% |
| 8 | 8 | 4 | 331.5 | 571 | 549 | 172.2% | 165.6% |
| 16 | 16 | 8 | 431.0 | 731 | 739 | 169.6% | 171.5% |
| 32 | 32 | 16 | 481.8 | 777 | 786 | 161.3% | 163.1% |

</details>

<details>
<summary>ITL p50 (ms)</summary>

| Requests resident/GPU | Baseline concurrency | HAMi and Fractions concurrency | ITL p50 Baseline (ms) | ITL p50 HAMi (ms) | ITL p50 Fractions (ms) | HAMi % of Baseline | Fractions % of Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 8.61 | 19.0 | 13.2 | 220.7% | 153.3% |
| 4 | 4 | 2 | 9.13 | 19.3 | 14.1 | 211.4% | 154.4% |
| 8 | 8 | 4 | 11.00 | 21.1 | 15.4 | 191.8% | 140.0% |
| 16 | 16 | 8 | 15.56 | 25.2 | 19.3 | 162.0% | 124.0% |
| 32 | 32 | 16 | 26.20 | 35.6 | 28.9 | 135.9% | 110.3% |

</details>

### Cost per 1M Output Tokens

This table compares the cost of serving 16 models when each model receives the same traffic. Serving 16 models with one model per GPU requires leasing two 8× A100 nodes, while GPU fractions serves all 16 on one node. The baseline is extrapolated from the 8-model run by doubling its throughput. Both configurations are priced at a normalized $1.00 per node-hour; multiply the $/1M columns by your node-hour price to get your cost. The percentage column does not depend on the price.

Unlike the tables above, rows here are matched on concurrency per model, not requests per GPU.

| Concurrency per model | Baseline tok/s (2 nodes) | Baseline $/1M | Fractions tok/s (1 node) | Fractions $/1M | Fractions cost vs Baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,782 | $0.3118 | 1,155 | $0.2405 | -22.9% |
| 2 | 3,640 | $0.1526 | 2,074 | $0.1339 | -12.2% |
| 4 | 6,866 | $0.0809 | 3,649 | $0.0761 | -5.9% |
| 8 | 10,740 | $0.0517 | 5,782 | $0.0480 | -7.1% |
| 16 | 14,868 | $0.0374 | 8,027 | $0.0346 | -7.4% |
| 32 | 18,268 | $0.0304 | 9,059 | $0.0307 | +0.8% |

$/1M is the hourly node cost divided by output tokens per hour, scaled to one million tokens. Only output tokens are counted. At light per-model load, a dedicated GPU sits mostly idle, and packing two models onto it recovers that idle capacity. At 32 concurrent requests per model, the dedicated GPUs are busy enough that the cost advantage disappears.

## Reading the Table

### Rows Are Matched on Load per GPU

Each row compares the configurations at the same number of requests resident on each GPU. A shared GPU hosts two models, so HAMi and GPU fractions at concurrency `c` match the baseline at concurrency `2c`.

Comparing rows at equal `c` instead would give the shared configurations twice the load of the baseline and make them look better than they are.

### Latency Columns

**TTFT p50** (time to first token) and **ITL p50** (inter-token latency) are per-request medians. Sharing always costs latency, because each model runs on only part of the GPU. Treat these columns as a check that latency is still acceptable for your service. GPU fractions matches or beats HAMi on ITL in every row, and ITL stays under 30 ms throughout.

### Throughput Columns

**tok/s per GPU** is output tokens per second per GPU, summed across all models on that GPU. The absolute values depend on this model, GPU, and prompt shape, so they do not carry over to your workload.

### Percentage Columns

**Fractions % of Baseline** answers the deployment question directly: when you put two models on one GPU instead of giving each its own, how much of that GPU's throughput do you keep?

Base decisions on the percentage columns for three reasons:

1. **They cancel out the setup.** Model size, GPU type, and sequence lengths shift all three configurations together. A ratio against a dedicated baseline at the same load isolates the cost of sharing, which makes it the number most likely to transfer to your workload.
2. **They map to a GPU-count decision.** At one request per model, GPU fractions keeps 63.5%: one GPU serves two distinct models at about 63% of the throughput a dedicated GPU would deliver, rising to 87.9% at 16 requests per model. For every pair of models that can tolerate that, you free a whole GPU. In this benchmark, 16 models run on 8 GPUs instead of 16.
3. **They separate how the GPU is shared from the fact that it is shared.** HAMi and GPU fractions both pay the cost of splitting a GPU. The 15 to 20 point gap between them is the gain from partitioning compute instead of time-slicing it. Under time-slicing, one model's kernels wait behind the other's. With dedicated SMs, each model's kernels start immediately.

## Why Sharing Costs Throughput

Each decode step reads the model's full weights from GPU memory, and at low load that read dominates the step time. One model serving two requests reads its weights once per step for both. Two models serving one request each read two sets of weights, which roughly doubles the memory traffic for the same output. At low load this duplication dominates, and GPU fractions keeps about 63% of dedicated throughput.

As load rises, each step does more compute per weight read, so the duplicated reads matter less. GPU fractions climbs to about 88% of dedicated throughput at 32 requests per GPU.

## Summary

Read across a row to confirm that latency meets your requirements, then use **Fractions % of Baseline** to decide whether giving up that share of throughput is worth one GPU saved per pair of models.

## Reproducing the Results

Each configuration has its own deployment guide. The manifests and benchmark scripts they use are in [`examples/gpu-sharing/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing).

| Configuration | Guide |
|---|---|
| Baseline | [Deploy the Baseline Experiment](deploy-baseline.md) |
| HAMi | [Deploy the KAI + HAMi Experiment](deploy-kai-hami.md) |
| GPU fractions | [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md), then [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md) |

Scripts, manifests, and results: [@marckarp](https://github.com/marckarp) and [@scheckerNV](https://github.com/scheckerNV).

## Appendix: Node Throughput at Equal Per-Model Concurrency

This table compares total node output throughput (tok/s) with every configuration at the same per-model concurrency. The shared configurations run 16 models on 8 GPUs, so each GPU carries twice as many in-flight requests as in the vanilla configuration at the same `c/model`.

| c/model | Vanilla (8 models) | HAMi (16 models) | HAMi % of vanilla | Fractions (16 models) | Fractions % of vanilla |
|---:|---:|---:|---:|---:|---:|
| 1 | 891 | 809 | 90.8% | 1,155 | 129.6% |
| 2 | 1,820 | 1,544 | 84.8% | 2,074 | 114.0% |
| 4 | 3,433 | 2,751 | 80.1% | 3,649 | 106.3% |
| 8 | 5,370 | 4,576 | 85.2% | 5,782 | 107.7% |
| 16 | 7,434 | 6,651 | 89.5% | 8,027 | 108.0% |
| 32 | 9,134 | 7,544 | 82.6% | 9,059 | 99.2% |
