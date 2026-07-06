---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Benchmarking with AIPerf
subtitle: Measure latency and throughput of a Dynamo deployment from the command line
---

[AIPerf](https://github.com/ai-dynamo/aiperf) is a standalone command-line tool for measuring generative AI inference performance. Point it at any OpenAI-compatible HTTP endpoint — a DynamoGraphDeployment frontend or an external service — and it reports latency, throughput, Time To First Token (TTFT), and inter-token latency, with real-time dashboards and automatic visualization. AIPerf is developed and versioned separately from NVIDIA Dynamo and is available on [PyPI](https://pypi.org/project/aiperf/); it is also pre-installed in Dynamo container images.

```bash
pip install aiperf
```

> [!NOTE]
> The `--model` parameter must match the model deployed at the endpoint.

## When to use it

Reach for AIPerf when you want to measure real performance against a running endpoint:

- Benchmark a DynamoGraphDeployment across concurrency levels to find its saturation point.
- Compare configurations — aggregated versus disaggregated, KV-aware routing on or off, or one backend against another.
- Validate that a deployment meets your latency and throughput SLA after sizing it with [AIConfigurator](aic.md).

## Run a single benchmark

With the frontend reachable at `http://localhost:8000`, send a fixed number of requests at a set concurrency:

```bash
aiperf profile \
    --model <your-model-name> \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 10 \
    --request-count 100 \
    --synthetic-input-tokens-mean 2000 \
    --output-tokens-mean 256
```

This writes results to `artifacts/` and prints a metrics summary to the console. To sweep concurrency for a Pareto analysis and visualize the results with `aiperf plot`, follow the full guide below.

## Where to go next

TODO(broken-link): ../benchmarks/benchmarking.md resolves to /dynamo/dev/benchmarks/benchmarking, which is not registered in docs/index.yml, so this link is broken. Register that page in the nav (or fix the target) to resolve. Pre-existing: features/disaggregated-serving/aiconfigurator.md and backends/vllm/vllm-examples.md link to it the same way.
- For concurrency sweeps, client-side versus in-cluster benchmarking, visualization, and advanced features (trace replay, arrival patterns, GPU telemetry), see the [Dynamo Benchmarking](../benchmarks/benchmarking.md) guide.
- To benchmark a supported model or feature from a known-good baseline, start from a [Dynamo Recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes).
- For the upstream command reference, run `aiperf profile --help` or see the [AIPerf docs](https://github.com/ai-dynamo/aiperf/tree/main/docs).
