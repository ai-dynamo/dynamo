---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Performance Analysis
subtitle: Choose the right tool for a performance question on a Kubernetes deployment
---

Dynamo ships several performance tools and they answer different questions. This page routes a question to the tool that answers it. For the method that applies whichever tool you pick, see [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md).

> [!NOTE]
> This page names tools, not flags. Each tool's options live in its own page.

## Start with the question

| Your question | Go to |
| --- | --- |
| Is this deployment slower than it should be? | [Check health first](#diagnosis-check-before-you-measure) |
| What throughput does this configuration deliver? | [Characterization](#characterization-produce-a-number) |
| How many GPUs, and which parallelism? | [Sizing](#sizing-choose-a-configuration) |
| Did my change make it slower? | [Regression](#regression-compare-two-configurations) |
| Where is the time going? | [Attribution](#attribution-find-where-time-goes) |

## Diagnosis: check before you measure

Start here whenever something is slower than expected. A benchmark run against a misconfigured deployment produces a number that describes the misconfiguration.

1. Confirm pods, model cache, and the frontend are healthy, and that a single request succeeds.
2. For disaggregated serving, confirm the KV transport is on the intended path. A silent fallback to a slower transport looks exactly like a performance problem.
3. Confirm the router is in the mode you intend. KV-aware routing degrades to approximate matching when workers do not publish KV events.
4. Check [Observability](observability.mdx) dashboards for queue depth and per-component latency. Prefill processing time is the usual first suspect in a disaggregated deployment.

Only once those pass is a measurement meaningful.

## Characterization: produce a number

To benchmark a deployed model, use [Benchmark a Kubernetes Deployment with AIPerf](benchmarking-with-aiperf.mdx). AIPerf sends load to the live endpoint and measures what a client observes.

For a supported model, the [recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) carry a `perf.yaml` that runs AIPerf in-cluster against a fixed trace, which is reproducible and avoids port-forward bandwidth limits. Deploy the recipe, then apply its `perf.yaml`.

Published cross-configuration results, and the provenance contract they follow, are under [Feature Benchmarks](../../recipes/feature-benchmarks/browse-all-benchmarks.mdx).

## Sizing: choose a configuration

To pick GPU count and parallelism before committing cluster time, use the [Dynamo Profiler](../auto-deployment/dynamo-profiler.md). It has two search modes: a fast simulation-backed pass that needs no GPUs, and a thorough pass that deploys candidates and benchmarks each one.

To explore a wider space — worker split, router configuration, cache capacity — replay a workload through [Simulation with DynoSim](simulation-with-dynosim/overview.md).

> [!WARNING]
> Simulation ranks candidates; it does not measure them. The default timing model is uncalibrated, and several paths are not modelled at all, including multimodal encoder compute. Rank with simulation, then validate the winner on hardware, and never publish a simulated figure as a measured one.

Once a configuration is deployed, [Advanced Performance Tuning](performance-tuning.md) covers the workload-specific parameters no tool selects for you.

## Regression: compare two configurations

Deploy both arms, benchmark each with AIPerf under an identical workload, and compare. The arms must differ in one variable.

The comparison is only as good as its protocol: interleave the arms, run each more than once, and compare medians. Read [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md) before quoting a difference, particularly if the difference is small.

## Attribution: find where time goes

| You want | Use |
| --- | --- |
| Per-request stage breakdown, including prefill wait and decode | Request tracing, then convert the trace to a timeline view. See [Request Traces](../../reference/observability/request-traces.mdx) |
| Which component along the path is slow | [Observability](observability.mdx) dashboards, backed by the [metrics catalog](../../reference/observability/metrics-catalog.mdx) |
| Whether the interconnect is the limit | Interconnect validation, before attributing anything to Dynamo |

CPU-level profiling of Dynamo processes is a development-host activity. See the CLI guide's Performance Analysis page.

## Related

- [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md) — preconditions, run protocol, reporting contract
- [Benchmark a Kubernetes Deployment with AIPerf](benchmarking-with-aiperf.mdx)
- [Advanced Performance Tuning](performance-tuning.md)
- [Simulation with DynoSim](simulation-with-dynosim/overview.md)
