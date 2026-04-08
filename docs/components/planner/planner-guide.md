---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Guide
---

The Dynamo Planner is an autoscaling controller that adjusts prefill and decode engine replica counts at runtime to meet latency SLAs. It reads traffic signals (Prometheus metrics or load predictor output) and engine performance profiles to decide when to scale up or down.

For a quick overview, see the [Planner overview](README.md). For architecture internals, see [Planner Design](../../design-docs/planner-design.md).

## Scaling Modes

The planner supports two scaling modes that can be used independently or together:

- **Throughput-based scaling** (`enable_throughput_scaling: true`): Uses pre-deployment engine performance data (from self-benchmark or profiler) and traffic prediction to plan capacity. Best for stable, predictable workloads.
- **Load-based scaling** (`enable_load_scaling: true`): Uses real-time ForwardPassMetrics (FPM) from the Dynamo event plane and online regression to make scaling decisions. Best for bursty or unpredictable traffic. Does not require pre-deployment data.

**When to use which:**

- Enable **throughput-based scaling** whenever pre-deployment performance data is available (via self-benchmark or profiler). It provides stable, prediction-based capacity planning.
- Enable **load-based scaling** when traffic is bursty. It reacts quickly to real-time load changes.
- Enable **both** for the best of both worlds: throughput-based provides a capacity floor, load-based handles bursts above it. When both are enabled, use a longer `throughput_adjustment_interval`.

## PlannerConfig Reference

The planner is configured via a `PlannerConfig` JSON/YAML object. When using the profiler, this is placed under the `features.planner` section of the DGDR spec:

```yaml
features:
  planner:
    enable_throughput_scaling: true
    enable_load_scaling: false
    pre_deployment_sweeping_mode: rapid
    mode: disagg
    backend: vllm
```

### Scaling Mode Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_throughput_scaling` | bool | `true` | Enable throughput-based scaling (requires pre-deployment performance data). |
| `enable_load_scaling` | bool | `false` | Enable load-based scaling. |

At least one scaling mode must be enabled.

### Pre-Deployment Sweeping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pre_deployment_sweeping_mode` | string | `rapid` | How to generate engine performance data: `rapid` (AIC simulation, ~30s), `thorough` (real GPUs, 2-4h), or `none` (skip). |

When throughput-based scaling is enabled, the planner needs engine performance data. At startup, it first tries to fetch self-benchmark results from the `get_perf_metrics` Dynamo endpoint (see PR #7779). If unavailable, it falls back to profiler-generated data (npz or JSON) at `profile_results_dir`. Both sources are converted to ForwardPassMetrics and fed into the FPM regression model.

### Throughput-Based Scaling Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `throughput_adjustment_interval` | int | `180` | Seconds between throughput-based scaling decisions. |
| `min_endpoint` | int | `1` | Minimum number of engine endpoints to maintain. |
| `max_gpu_budget` | int | `8` | Maximum total GPUs the planner may allocate. |
| `ttft` | float | `500.0` | TTFT SLA target (ms) for scaling decisions. |
| `itl` | float | `50.0` | ITL SLA target (ms) for scaling decisions. |

### Load-Based Scaling Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_adjustment_interval` | int | `5` | Seconds between FPM regression updates and load-based scaling decisions. Even when only throughput scaling is enabled, live FPM observations are fed into the regression at this interval. Must be shorter than `throughput_adjustment_interval`. |
| `max_num_fpm_samples` | int | `64` | Maximum retained FPM observations for regression. |
| `fpm_sample_bucket_size` | int | `16` | Number of buckets for observation retirement (must be a perfect square). |
| `load_scaling_down_sensitivity` | int | `80` | Scale-down sensitivity 0–100 (0=never, 100=aggressive). |
| `load_metric_samples` | int | `10` | Number of metric samples to collect per decision. |
| `load_min_observations` | int | `5` | Minimum observations before making scaling decisions. |

### General Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `disagg` | Planner mode: `disagg`, `prefill`, `decode`, or `agg`. |
| `backend` | string | `vllm` | Backend: `vllm`, `sglang`, `trtllm`, or `mocker`. |
| `environment` | string | `kubernetes` | Runtime environment: `kubernetes`, `virtual`, or `global-planner`. |
| `namespace` | string | env `DYN_NAMESPACE` | Kubernetes namespace for the deployment. |

### Traffic Prediction Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_predictor` | string | `arima` | Prediction method: `constant`, `arima`, `kalman`, or `prophet`. |
| `load_predictor_log1p` | bool | `false` | Apply log1p transform to load data before prediction. |
| `prophet_window_size` | int | `50` | Window size (seconds) for Prophet predictor. |
| `load_predictor_warmup_trace` | string | `null` | Path to a warmup trace file for bootstrapping predictions. |

### Kalman Filter Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kalman_q_level` | float | `1.0` | Process noise for level component. |
| `kalman_q_trend` | float | `0.1` | Process noise for trend component. |
| `kalman_r` | float | `10.0` | Measurement noise. |
| `kalman_min_points` | int | `5` | Minimum data points before Kalman predictions activate. |

## Integration with Profiler

When the profiler runs with planner enabled, it:

1. Selects the best prefill and decode engine configurations
2. Generates engine performance data (prefill TTFT vs ISL, decode ITL vs KV-cache utilization)
3. Saves the `PlannerConfig` and performance data into separate Kubernetes ConfigMaps
4. Adds the planner service to the generated DGD, configured to read from those ConfigMaps

The planner receives its config via `--config /path/to/planner_config.json` which is mounted from the `planner-config-XXXX` ConfigMap. Profiling data is mounted from the `planner-profile-data-XXXX` ConfigMap.

See the [Profiler Guide](../profiler/profiler-guide.md) for the full profiling workflow and how to configure pre-deployment sweeping.

## Hierarchical Deployments

If you want one public endpoint for a model but multiple private DGDs optimized for different request classes, use a hierarchical deployment:

- one control DGD with `Frontend`, `GlobalRouter`, and `GlobalPlanner`
- one or more prefill pool DGDs
- one or more decode pool DGDs

In the current workflow, run profiling independently for each intended pool, then compose the final control DGD plus pool DGDs manually. See the [Global Planner Guide](global-planner.md).

## See Also

- [Planner overview](README.md) — Why LLM inference needs a different autoscaler
- [Planner Design](../../design-docs/planner-design.md) — Architecture and algorithm internals
- [Planner Examples](planner-examples.md) — DGDR YAML examples, sample configurations, advanced patterns
- [Global Planner Guide](global-planner.md) — Multi-DGD coordination, shared GPU budgets, single-endpoint multi-pool deployments
- [Profiler Guide](../profiler/profiler-guide.md) — How profiling data is generated
