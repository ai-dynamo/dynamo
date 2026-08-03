---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Performance Analysis
subtitle: Choose the right tool for a performance question on a development host
---

Dynamo ships several performance tools and they answer different questions. This page routes a question to the tool that answers it, for work on a single machine. For the method that applies whichever tool you pick, see [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md).

For measuring a Kubernetes deployment instead, see the Kubernetes guide's Performance Analysis page.

> [!NOTE]
> This page names tools, not flags. Each tool's options live in its own page.

## Start with the question

| Your question | Go to |
| --- | --- |
| Did my code change make Dynamo slower? | [Regression](#regression-compare-two-revisions) |
| Where does the CPU time go? | [Attribution](#attribution-find-where-time-goes) |
| How does this configuration behave under a workload? | [Characterization](#characterization-produce-a-number) |
| Which configuration should I deploy? | [Sizing](#sizing-choose-a-configuration) |

## Regression: compare two revisions

First decide what your change can affect, because that decides whether you need GPUs at all.

**Routing, scheduling, admission, or cache-accounting changes.** Replay a workload offline through [Simulation with DynoSim](simulation-with-dynosim/overview.md). Offline replay runs the production router and scheduler against simulated engines on a logical clock, so it is deterministic, needs no GPUs, and runs faster than real time. Determinism is the point: it makes byte-for-byte comparison between revisions possible, which is a far stronger signal than a timing difference.

**Frontend, tokenizer, or request-dispatch changes.** Run the frontend against mock workers so the engine is not a variable. The [frontend benchmark harness](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/frontend) sweeps concurrency and input length and collects profiling captures alongside the load.

**Anything touching the engine or the transport.** Neither of the above sees it. Simulation replaces the transport with an event queue, and mock workers replace the engine. Measure against a real backend.

Whichever you pick, the comparison is only as good as its protocol. Interleave the arms, run each more than once, compare medians, and change one variable. See [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md) before quoting a difference.

## Attribution: find where time goes

| You want | Use |
| --- | --- |
| Which functions burn CPU | An on-CPU profile and flame graph, taken while load runs |
| What blocked threads wait on | An off-CPU profile. Asynchronous runtimes park idle worker threads, so most parked time is benign; look for application-level lock frames |
| Per-request stage breakdown | Request tracing, converted to a timeline. See [Request Traces](../../reference/observability/request-traces.mdx) |
| Which internal stage is slow | Frontend stage-duration and queue-depth metrics from the [metrics catalog](../../reference/observability/metrics-catalog.mdx) |
| GPU and kernel timeline | Nsight Systems, with the NVTX build feature enabled |

Build with the `profiling` Cargo profile for these. It inherits release optimization but keeps debug symbols, without which release stacks are unreadable. Profiling scripts live alongside the [frontend benchmark harness](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/frontend).

> [!IMPORTANT]
> Off-CPU profiling requires root, and release builds omit frame pointers, so stack walking needs DWARF unwinding. A profile that looks empty is usually one of these two, not an idle process.

## Characterization: produce a number

To measure a local endpoint, use [Benchmark with AIPerf](benchmarking-with-aiperf.mdx).

Without GPUs, run the frontend against the [mocker engine](simulation-with-dynosim/mocker-live-simulation.mdx) to exercise the full request path with simulated generation. This characterizes Dynamo's own overhead, not model serving performance.

> [!IMPORTANT]
> A closed-loop client at fixed concurrency measures concurrency divided by latency, not maximum throughput. Idle server cores usually mean the system is latency-bound rather than slow. Sweep concurrency to find saturation, and watch that the load generator itself is not the bottleneck.

## Sizing: choose a configuration

Use the [profiler](../../developer-guide/knowledge-base/modular-components/profiler/overview.md) to pick GPU count and parallelism, and [DynoSim sweeps](simulation-with-dynosim/dynosim-sweeps.mdx) to explore worker split, router configuration, and cache capacity.

> [!WARNING]
> Simulation ranks candidates; it does not measure them. The default timing model is uncalibrated, and several paths are not modelled at all, including multimodal encoder compute. Rank with simulation, then validate the winner on hardware, and never publish a simulated figure as a measured one. See [Simulation Model](simulation-with-dynosim/simulation-model.md) for the fidelity boundaries.

## Related

- [Performance Analysis Method](../../developer-guide/knowledge-base/concepts/performance-analysis-method.md) — preconditions, run protocol, reporting contract
- [Benchmark with AIPerf](benchmarking-with-aiperf.mdx)
- [Simulation with DynoSim](simulation-with-dynosim/overview.md)
- [Observability](observability.mdx)
