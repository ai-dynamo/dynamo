---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Guide
---

The Dynamo Planner is an autoscaling controller that adjusts prefill and decode engine replica counts at runtime to meet latency SLAs. It reads traffic signals (Prometheus metrics or load predictor output) and engine performance profiles to decide when to scale up or down.

For a quick overview, see the [Planner overview](README.md). For architecture internals, see [Planner Design](../../design-docs/planner-design.md).

## Scaling Modes

The planner supports three optimization targets that determine how scaling decisions are made:

- **`throughput`** (default): Uses static thresholds on queue depth and KV cache utilization. No SLA targets or profiling needed. Works out of the box.
- **`latency`**: Same approach as `throughput` but with more aggressive thresholds — scales up earlier and tolerates less queuing. Ideal for latency-sensitive workloads.
- **`sla`**: Uses regression-based performance models with specific TTFT/ITL targets. Supports both throughput-based (predictive) and load-based (reactive) scaling modes. For advanced users who need precise SLA control.

**When to use which:**

- Start with **`throughput`** (the default) — it works immediately with no configuration.
- Switch to **`latency`** if your workload has strict latency requirements and you prefer to over-provision rather than queue.
- Use **`sla`** when you have pre-deployment profiling data and want to target specific TTFT/ITL values.

## PlannerConfig Reference

The planner is configured via a `PlannerConfig` JSON/YAML object. When using the profiler, this is placed under the `features.planner` section of the DGDR spec:

```yaml
features:
  planner:
    mode: disagg
    backend: vllm
    # optimization_target defaults to "throughput" — works out of the box
```

For SLA-based scaling:

```yaml
features:
  planner:
    optimization_target: sla
    enable_throughput_scaling: true
    enable_load_scaling: false
    pre_deployment_sweeping_mode: rapid
    mode: disagg
    backend: vllm
```

### Optimization Target

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimization_target` | string | `throughput` | `throughput`: scale based on queue/utilization thresholds. `latency`: aggressive low-latency thresholds. `sla`: regression-based scaling with ttft/itl targets. |

When `optimization_target` is `throughput` or `latency`, load-based scaling is automatically enabled and throughput-based scaling is disabled. The `ttft`/`itl` fields are ignored.

### Scaling Mode Fields (SLA mode)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_throughput_scaling` | bool | `true` | Enable throughput-based scaling (requires pre-deployment performance data). Only used when `optimization_target: sla`. |
| `enable_load_scaling` | bool | `false` | Enable load-based scaling. Only used when `optimization_target: sla`. |

At least one scaling mode must be enabled when using `optimization_target: sla`.

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

### Diagnostics Reports

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `report_interval_hours` | float or `null` | `24.0` | Generate an HTML diagnostics report every N hours (simulated time). Set to `null` to disable periodic report generation. |
| `report_output_dir` | string | `./planner_reports` | Directory for HTML diagnostics reports. |
| `live_dashboard_port` | int | `8080` | Port for the live diagnostics dashboard HTTP server. Set to `0` to disable. When enabled, visit `http://host:port/` to view a real-time Plotly report of accumulated snapshots. |

The same diagnostic signals surfaced in these reports are also exported as Prometheus metrics under the `dynamo_planner_*` prefix—for example estimated TTFT/ITL (`dynamo_planner_estimated_ttft_ms`, `dynamo_planner_estimated_itl_ms`), per-engine capacity and FPM queue depths, and load/throughput scaling decision enums.

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
