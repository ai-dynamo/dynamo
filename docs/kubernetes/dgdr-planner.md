---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tune the Planner
subtitle: Pick the right Planner optimization target and scaling knobs for your workload, and validate the choice before it scales GPUs.
---

The [Auto Deploy with DGDR](dgdr-guide.md) guide shows how to turn the Planner on with a few lines under `features.planner`. This page picks up where that leaves off: which Planner knobs actually change behavior, and when to reach for each one.

The Planner exposes a long list of fields. Most workloads only need a handful. This guide covers the ones that matter — the optimization target, the two SLA scaling modes, and the guardrails that keep scaling safe — and points to the [PlannerConfig reference](../components/planner/planner-guide.md#plannerconfig-reference) for the rest. Set every field under `spec.features.planner` in the DGDR; DGDR passes the object through to the Planner service.

> [!NOTE]
> This is a how-to for tuning a Planner you have already enabled. For why LLM inference needs a purpose-built autoscaler and the prefill/decode scaling model, read the [Planner overview](../components/planner/README.md) first. For the exhaustive field table, see the [Planner Guide](../components/planner/planner-guide.md).

## Two levels of tuning

Planner tuning happens at two levels, and it helps to keep them separate:

1. **The optimization target** (`optimization_target`) picks the overall strategy — how the Planner decides it needs to scale at all. This is the first knob everyone touches.
2. **The SLA scaling modes** (`enable_throughput_scaling`, `enable_load_scaling`) tune *how* the Planner scales, but only when the target is `sla`. They are ignored for every other target.

> [!WARNING]
> `optimization_target: throughput` and `enable_throughput_scaling` are **different knobs** despite the shared word. The first is a simple, zero-config target that scales on queue depth and KV-cache utilization. The second is a predictive scaling *mechanism* that only runs under `optimization_target: sla`. Setting `enable_throughput_scaling` while the target is `throughput` does nothing. Level 1 chooses the strategy; level 2 only applies once you have chosen `sla`.

## Choose an optimization target

`optimization_target` is the knob to set first. It has four values, in rough order of how much setup they need:

| Target | How it scales | Needs SLA targets? | Needs profiling? |
|---|---|:---:|:---:|
| `throughput` (default) | Static thresholds on queue depth and KV-cache utilization | No | No |
| `latency` | Same as `throughput`, but scales up earlier and tolerates less queuing | No | No |
| `load` | User-defined prefill queue-token and decode KV-utilization thresholds | No | No |
| `sla` | Rust engine performance model targeting specific TTFT/ITL values | Yes | Recommended |

**Reach for each when:**

- **`throughput`** — Start here. It works out of the box with no SLA targets and no profiling data, and it's the right default when you don't yet have latency requirements to hit.
- **`latency`** — Switch when your workload is latency-sensitive and you'd rather over-provision than let requests queue. It uses the same thresholds as `throughput` with a more aggressive trigger.
- **`load`** — Use when you want to drive scaling directly from thresholds you set — prefill queue tokens and decode KV-cache utilization — instead of the built-in ones. Good when you understand your engine's saturation points and want explicit control without committing to full SLA modeling.
- **`sla`** — Use when you need to hold a specific TTFT and ITL. This is the only target that models engine performance to translate "keep P95 TTFT under 500 ms" into replica counts. It's the most powerful and the most involved — the rest of this page is mostly about tuning it.

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      optimization_target: latency   # throughput | latency | load | sla
```

> [!NOTE]
> For `throughput`, `latency`, and `load`, the Planner enables load-based scaling automatically and ignores `ttft_ms`/`itl_ms`. Those SLA targets and the two `enable_*_scaling` knobs below only take effect under `optimization_target: sla`.

## Tune SLA mode: throughput vs. load scaling

Under `optimization_target: sla`, two scaling mechanisms decide when to change replica counts. You enable them independently, and **at least one must be on.**

| | Throughput-based (`enable_throughput_scaling`) | Load-based (`enable_load_scaling`) |
|---|---|---|
| **Default** | `true` | `false` |
| **How it decides** | Predicts traffic, computes the replica count needed to meet SLA | Reacts to live ForwardPassMetrics from running engines |
| **Cadence** | Long — `throughput_adjustment_interval_seconds` (default `180`) | Short — `load_adjustment_interval_seconds` (default `5`) |
| **Role** | Sets a stable capacity floor | Handles bursts above that floor |
| **Best for** | Predictable traffic, steady-state planning | Bursty or hard-to-predict traffic |

**Reach for each when:**

- **Throughput-based scaling** gives you stable, prediction-driven capacity planning. Keep it on (the default) when traffic is reasonably predictable and you want the Planner to hold a sensible baseline rather than chase every spike.
- **Load-based scaling** reacts fast — it reads per-iteration engine metrics and adjusts on a short interval, so it catches bursts that prediction misses. Turn it on when traffic is spiky.
- **Both together** is the recommended production setup: throughput-based scaling sets the long-term floor, and load-based scaling adjusts above it in real time. When you enable both, keep `throughput_adjustment_interval_seconds` well above `load_adjustment_interval_seconds` so the two mechanisms don't fight.

```yaml
spec:
  features:
    planner:
      optimization_target: sla
      enable_throughput_scaling: true
      enable_load_scaling: true
      ttft_ms: 500.0
      itl_ms: 50.0
      pre_deployment_sweeping_mode: rapid
```

### Warm up the performance model

SLA mode relies on a performance model to translate latency targets into replica counts. How fast it becomes accurate depends on `pre_deployment_sweeping_mode`:

| Value | What it does | Cost |
|---|---|---|
| `rapid` (default) | Generates bootstrap data with AIC simulation | ~30s, no GPUs |
| `thorough` | Benchmarks candidate configs on real GPUs | 2–4h, real GPUs |
| `none` | Skips bootstrap; the model warms from live traffic | Free, but slower to converge |

Use `rapid` for most cases. Reach for `thorough` when you need measured accuracy for a production deployment or your hardware isn't covered by AIC. With `none`, the Planner still starts, but throughput decisions report `model_not_ready` until enough live metrics accumulate — acceptable when load-based scaling is carrying real-time response and you're comfortable with a slower warmup.

> [!NOTE]
> Reaching SLA mode through a DGDR is the fastest path because the profiler generates this bootstrap data for you. See [Auto Deploy with DGDR — Enable the Planner](dgdr-guide.md#enable-the-planner-for-runtime-autoscaling) for the profiling workflow.

## Set the guardrails

Whatever target you choose, a few knobs bound what the Planner is allowed to do. Set these before you let it scale unattended:

| Field | Default | What it protects |
|---|---|---|
| `min_endpoint` | `1` | Floor on replicas per worker type. Raise it above your steady-state traffic floor so the Planner never scales below what you always need. |
| `max_gpu_budget` | `8` | Ceiling on total GPUs the Planner may allocate. Your cost cap — scaling never exceeds it. |
| `load_scaling_down_sensitivity` | `80` | How eagerly load-based scaling removes replicas (0 = never, 100 = aggressive). Lower it to scale down less often. |

`min_endpoint` and `load_scaling_down_sensitivity` matter most because of how scale-down works: when the Planner removes a worker, it terminates it **without draining in-flight requests**, so requests mid-prefill on that worker fail. In disaggregated deployments this can also fail decode workers waiting on a KV-cache transfer from the terminated prefill worker. To reduce the blast radius, set `min_endpoint` above your steady-state floor and lower `load_scaling_down_sensitivity` so scale-down fires less often. See [Current Limitations](../components/planner/README.md#current-limitations) for the full caveat.

## Validate before you scale for real

Before letting a new configuration move GPUs, run it in **advisory mode**. The Planner observes traffic, computes recommended replica counts, logs them, and exports them as metrics — but never actually scales.

```yaml
spec:
  features:
    planner:
      optimization_target: sla
      enable_throughput_scaling: true
      ttft_ms: 500.0
      itl_ms: 50.0
      advisory: true
```

Watch the recommended counts against your actual traffic — `dynamo_planner_predicted_num_prefill_replicas` and `dynamo_planner_predicted_num_decode_replicas`, or the Replica Counts plot in the [diagnostics reports](../components/planner/README.md#html-diagnostics-reports), which overlays recommendations on actual replicas. Use advisory mode when validating new SLA targets, evaluating a config change, or confirming how the Planner would react to production traffic. Once the recommendations look right, drop `advisory: true` and let it scale.

## Go deeper

This page covers the knobs most workloads touch. For everything it skips:

| Topic | Where |
|---|---|
| Full PlannerConfig field table (load predictors, plugin pipeline, diagnostics) | [Planner Guide](../components/planner/planner-guide.md#plannerconfig-reference) |
| Load predictor tuning (ARIMA, Prophet, Kalman) | [Planner Examples](../components/planner/planner-examples.md#custom-load-predictors) |
| DGDR `features.planner` passthrough and field reference | [DGDR Reference — Planner](dgdr-reference.mdx#planner) |
| Copy-ready DGDR manifests | [DGDR Examples](dgdr-examples.md) |
| Multi-DGD coordination and shared GPU budgets | [Global Planner Guide](../components/planner/global-planner.md) |
| Architecture and algorithm internals | [Planner Design](../design-docs/planner-design.md) |
