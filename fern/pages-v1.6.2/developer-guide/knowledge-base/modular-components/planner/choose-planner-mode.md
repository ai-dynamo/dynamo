---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Choose a Planner Mode
subtitle: Select a deployment topology, optimization target, scaling method, and runtime environment for the Dynamo Planner.
---

The Planner has several settings that describe different decisions. Choose them in this order so that topology, scaling policy, and runtime behavior stay aligned.

| Decision | Configuration | Question |
|----------|---------------|----------|
| Deployment topology | `mode` | Which worker roles does this Planner scale? |
| Optimization target | `optimization_target` | What outcome should trigger scaling? |
| Scaling method | `enable_throughput_scaling`, `enable_load_scaling` | How should the Planner calculate replica recommendations for an `sla` target? |
| Runtime environment | `environment` | Should the Planner apply changes locally, delegate them, or simulate them? |

## Recommended Starting Points

| Requirement | Optimization Target | Scaling Method |
|-------------|---------------------|----------------|
| No specific latency SLA | `throughput` by default; `latency` when shorter queues matter more than GPU efficiency | Load-based, enabled automatically |
| Specific TTFT and ITL SLA | `sla` | Enable throughput-based and load-based scaling together |

Use `load` instead of `throughput` or `latency` only when you want to supply the prefill queue-token and decode KV-utilization thresholds yourself.

## Choose the Deployment Topology

Set `mode` to match the worker topology that the Planner controls.

| `mode` | Worker Topology | Use When |
|--------|-----------------|----------|
| `disagg` (default) | Separate prefill and decode workers | The deployment uses disaggregated serving and each worker role must scale independently. |
| `agg` | One worker performs prefill and decode | The deployment uses aggregated serving and needs one replica count. |
| `prefill` | Prefill workers only | The Planner controls a prefill-only pool, typically in a multi-DGD deployment. |
| `decode` | Decode workers only | The Planner controls a decode-only pool, typically in a multi-DGD deployment. |

For a single DGD, use `disagg` or `agg` to match the deployment. Use `prefill` and `decode` for independently managed pools. To coordinate multiple DGDs or expose multiple pools through one endpoint, see the [Global Planner Guide](global-planner-guide.md).

## Choose the Optimization Target

Choose the target based on whether you have specific Time To First Token (TTFT) and Inter-Token Latency (ITL) requirements.

| `optimization_target` | Use When | Scaling Behavior |
|-----------------------|----------|------------------|
| `throughput` (default) | You do not have a specific latency SLA and want the default balance of throughput and GPU use. | Uses built-in queue-depth and KV-utilization thresholds. |
| `latency` | You do not have a specific latency SLA but prefer shorter queues and earlier scale-up. | Uses more aggressive built-in thresholds. |
| `load` | You know the engine saturation points and want to configure thresholds directly. | Uses your prefill queue-token and decode KV-utilization thresholds. |
| `sla` | You must target specific TTFT and ITL values. | Uses performance-model estimates and lets you select the scaling methods. |

The `throughput`, `latency`, and `load` targets always use load-based scaling. They disable throughput-based scaling and ignore `enable_throughput_scaling` and `enable_load_scaling`.

## Choose Scaling Methods for an SLA

The `sla` target is the only target that lets you select scaling methods. Enable at least one.

| Scaling Method | Use When | Default |
|----------------|----------|:-------:|
| Throughput-based | Traffic is predictable enough for forecasting and you want a stable capacity floor. | On |
| Load-based | Traffic is bursty or difficult to predict and needs faster reactive adjustments. | Off |
| Both | You want prediction-based baseline capacity and burst response. | Recommended |

For most SLA-driven production deployments, enable both methods:

```yaml
features:
  planner:
    mode: disagg
    backend: vllm
    optimization_target: sla
    enable_throughput_scaling: true
    enable_load_scaling: true
    ttft_ms: 500.0
    itl_ms: 50.0
```

Keep `throughput_adjustment_interval_seconds` longer than `load_adjustment_interval_seconds` when both methods are enabled. The throughput-based method sets the capacity floor, then the load-based method adjusts above it.

## Choose the Runtime Environment

| `environment` | Behavior | Use When |
|---------------|----------|----------|
| `kubernetes` (default) | Applies replica changes to the local DynamoGraphDeployment (DGD). | One Planner controls one DGD on Kubernetes. |
| `global-planner` | Sends scale requests to a GlobalPlanner. | Multiple DGDs need centralized policy, authorization, or a shared GPU budget. |
| `virtual` | Applies changes through the VirtualConnector. | You are simulating, replaying, or integrating a non-Kubernetes scaling environment. |

Set `global_planner_namespace` when `environment` is `global-planner`. See the [Global Planner Guide](global-planner-guide.md) for the control DGD, pool-local Planner, and routing requirements.

## Check Dependencies

| Choice | Required | Optional or Recommended |
|--------|----------|-------------------------|
| Any load-based scaling | A supported backend that emits ForwardPassMetrics (FPM) through the Dynamo event plane | KV-aware routing is optional. |
| `throughput` or `latency` target | FPM from the backend | No SLA values, Prometheus traffic queries, or profiling data are required. |
| `load` target | FPM plus the queue-token or KV-utilization thresholds for the active worker roles | No profiling data is required. |
| `sla` target | TTFT and ITL values, Prometheus, and a supported performance-model path | Native AIConfigurator estimates or bootstrap profiling data reduce warmup time; live FPM can warm the regression fallback. |
| `global-planner` environment | Dynamo Kubernetes Platform, GlobalPlanner, pool-local routers, and Prometheus scraping router metrics | Profile each intended pool before composing a multi-pool deployment. |

The KV router is not required for load-based scaling. The Planner receives engine load through FPM regardless of the routing strategy. For backend-specific FPM requirements, see [Current Limitations](overview.md#current-limitations). For profiler bootstrap options, see the [Profiler Guide](../profiler/profiler-guide.md).

## Validate Before Applying Changes

Set `advisory: true` to calculate, log, and export recommendations without changing replica counts. Use advisory mode when introducing an SLA, changing targets, or validating a new workload.

```yaml
features:
  planner:
    optimization_target: sla
    enable_throughput_scaling: true
    enable_load_scaling: true
    advisory: true
```

Set replica floors and `max_gpu_budget` before disabling advisory mode. See [Tune the Planner](../../../../kubernetes/auto-deployment/dynamo-planner.mdx) for the deployment workflow and the [Planner Configuration reference](../../../../reference/components/planner-configuration.mdx) for every field and validation rule.
