---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Guide
---

<p align="left">
  <a href="./planner-guide.zh-CN.md" hreflang="zh-CN"><img src="../../assets/img/readme-zh-cn-link.svg" alt="简体中文" height="28" /></a>
</p>

The Dynamo Planner is an autoscaling controller that adjusts prefill and decode engine replica counts at runtime to meet latency SLAs. It reads traffic signals (Prometheus metrics or load predictor output) and engine performance models to decide when to scale up or down.

For a quick overview, see the [Planner overview](README.md). For architecture internals, see [Planner Design](../../design-docs/planner-design.md).

## Scaling Modes

The planner supports four optimization targets that determine how scaling decisions are made:

- **`throughput`** (default): Uses static thresholds on queue depth and KV cache utilization. No SLA targets or profiling needed. Works out of the box.
- **`latency`**: Same approach as `throughput` but with more aggressive thresholds — scales up earlier and tolerates less queuing. Ideal for latency-sensitive workloads.
- **`load`**: Uses user-defined prefill queue token thresholds and decode KV utilization thresholds for reactive load-based scaling.
- **`sla`**: Uses the Rust engine performance shim with native AIC estimates when available, plus online FPM tuning or FPM regression fallback, to target specific TTFT/ITL values. Supports both throughput-based (predictive) and load-based (reactive) scaling modes. For advanced users who need precise SLA control.

**When to use which:**

- Start with **`throughput`** (the default) — it works immediately with no configuration.
- Switch to **`latency`** if your workload has strict latency requirements and you prefer to over-provision rather than queue.
- Use **`load`** when you want direct control through prefill queue and decode KV utilization thresholds.
- Use **`sla`** when you want to target specific TTFT/ITL values with native AIC estimates, optional bootstrap profiling data, or live FPM warmup.

## PlannerConfig Reference

The planner is configured via a `PlannerConfig` JSON/YAML object. When using the
profiler, place this under `spec.features.planner` in the DGDR spec. Any
PlannerConfig field listed below can be set there; DGDR passes that object to
the Planner service for validation.

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      # optimization_target defaults to "throughput" — works out of the box
```

For SLA-based scaling:

```yaml
spec:
  features:
    planner:
      optimization_target: sla
      enable_throughput_scaling: true
      enable_load_scaling: false
      pre_deployment_sweeping_mode: rapid
      mode: disagg
      backend: vllm
```

To evaluate Planner behavior without changing replica counts, turn on advisory mode:

```yaml
spec:
  features:
    planner:
      advisory: true
```

Advisory mode is suggestion-only. The Planner computes recommended replica counts, logs them, exports them as diagnostics, and shows them in HTML reports. The recommendations are not applied as scaling decisions: the Planner does not execute scaling actions or change the deployment.

### Optimization Target

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimization_target` | string | `throughput` | `throughput`: scale based on queue/utilization thresholds. `latency`: aggressive low-latency thresholds. `load`: user-defined prefill queue and decode KV utilization thresholds. `sla`: Rust engine perf model scaling with ttft_ms/itl_ms targets. |

When `optimization_target` is `throughput`, `latency`, or `load`, load-based scaling is automatically enabled and throughput-based scaling is disabled. The `ttft_ms`/`itl_ms` fields are ignored.

### Scaling Mode Fields (SLA mode)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_throughput_scaling` | bool | `true` | Enable throughput-based scaling. Only used when `optimization_target: sla`. |
| `enable_load_scaling` | bool | `false` | Enable load-based scaling. Only used when `optimization_target: sla`. |

At least one scaling mode must be enabled when using `optimization_target: sla`.

### Pre-Deployment Sweeping

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pre_deployment_sweeping_mode` | string | `rapid` | How to generate optional bootstrap performance data: `rapid` (AIC simulation, ~30s), `thorough` (real GPUs, 2-4h), or `none` (skip). |

SLA mode uses the Rust engine performance shim. If `aic_perf_model` is present, the planner initializes the shim with native AIC model identity and engine limits. Unsupported native AIC configs automatically fall back to observed-FPM regression in the shim. If `aic_perf_model` is absent, the shim starts as an FPM regression model and becomes ready after enough self-benchmark or live FPM observations.

At startup, the planner always tries to fetch self-benchmark results from the `get_perf_metrics` Dynamo endpoint. If unavailable, it falls back to rapid-mode AIC interpolation data or profiler-generated data (npz or JSON) at `profile_results_dir` when configured. These sources are converted to ForwardPassMetrics and used to tune or bootstrap the perf model. With `pre_deployment_sweeping_mode: none`, the planner can still start; throughput decisions report `model_not_ready` until native AIC is available or enough live FPMs have warmed the regression fallback.

Manual native AIC perf-model config:

```yaml
features:
  planner:
    optimization_target: sla
    aic_perf_model:
      hf_id: nvidia/Llama-3.1-8B-Instruct-FP8
      system: h200_sxm
      backend: vllm
      prefill_pick: {tp: 1, pp: 1, dp: 1, moe_tp: 1, moe_ep: 1}
      decode_pick: {tp: 1, pp: 1, dp: 1, moe_tp: 1, moe_ep: 1}
```

### Throughput-Based Scaling Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `throughput_adjustment_interval_seconds` | int | `180` | Seconds between throughput-based scaling decisions. |
| `throughput_metrics_source` | string | `frontend` | Prometheus traffic source for throughput scaling: `frontend` reads `dynamo_frontend_*` metrics from the public Frontend; `router` reads `dynamo_component_router_*` metrics from a LocalRouter. Use `router` for pool-local Planner in GlobalPlanner deployments. |
| `min_endpoint` | int | `1` | Minimum number of engine endpoints to maintain. |
| `max_gpu_budget` | int | `8` | Maximum total GPUs the planner may allocate. |
| `ttft_ms` | float | `500.0` | TTFT SLA target (ms) for scaling decisions. |
| `itl_ms` | float | `50.0` | ITL SLA target (ms) for scaling decisions. |

### Load-Based Scaling Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_adjustment_interval_seconds` | int | `5` | Seconds between FPM tuning updates and load-based scaling decisions. Even when only throughput scaling is enabled, live FPM observations are fed into the perf model at this interval. Must be shorter than `throughput_adjustment_interval_seconds`. |
| `max_num_fpm_samples` | int | `64` | Maximum retained FPM observations for online tuning or regression. |
| `fpm_sample_bucket_size` | int | `16` | Number of buckets for observation retirement (must be a perfect square). |
| `load_scaling_down_sensitivity` | int | `80` | Scale-down sensitivity 0–100 (0=never, 100=aggressive). |
| `load_min_observations` | int | `5` | Minimum observations before making scaling decisions. |

### General Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `disagg` | Planner mode: `disagg`, `prefill`, `decode`, or `agg`. |
| `backend` | string | `vllm` | Backend: `vllm`, `sglang`, `trtllm`, or `mocker`. |
| `environment` | string | `kubernetes` | Runtime environment: `kubernetes`, `virtual`, or `global-planner`. |
| `namespace` | string | env `DYN_NAMESPACE` | Kubernetes namespace for the deployment. |
| `advisory` | bool | `false` | Suggestion-only mode. Compute, log, export, and report recommended replica counts without executing scaling actions or changing the deployment. |

### Traffic Prediction Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_predictor` | string | `arima` | Prediction method for request count, ISL, and OSL: `constant`, `arima`, `kalman`, or `prophet`. Runtime metadata such as KV hit rate and speculative decode accept length uses the latest valid observation instead. |
| `load_predictor_log1p` | bool | `false` | Apply log1p transform to predicted request count, ISL, and OSL data. |
| `prophet_window_size` | int | `50` | Window size (seconds) for Prophet predictor. |
| `load_predictor_warmup_trace` | string | `null` | Path to a warmup trace file for bootstrapping predictions. |

KV hit rate and speculative decode accept length are runtime engine/router signals, not traffic shape. The planner stores the latest valid observation for each signal and reuses it until a newer valid value arrives. On cold start, missing KV hit rate means no prefix-cache discount, and missing accept length means `1.0`.

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

The same diagnostic signals surfaced in these reports are also exported as Prometheus metrics under the `dynamo_planner_*` prefix—for example estimated TTFT/ITL (`dynamo_planner_estimated_ttft_ms`, `dynamo_planner_estimated_itl_ms`), recommended replica counts (`dynamo_planner_predicted_num_prefill_replicas`, `dynamo_planner_predicted_num_decode_replicas`), per-engine capacity and FPM queue depths, and load/throughput scaling decision enums.

The Replica Counts plot overlays actual prefill/decode replicas with discrete recommendation markers for the Planner's recommended prefill/decode replicas. When `advisory: true`, these recommended counts are suggestions only; the Planner records what it would do without applying the change.

### Scheduling / plugin pipeline

The planner runs through the builtin plugin pipeline by default. The base
pipeline cadence lives under the `scheduling` sub-tree of `PlannerConfig`;
plugin registration, transport, and auth settings live under
`plugin_registration`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scheduling.scale_interval_seconds` | float | gcd of enabled builtin intervals | Base pipeline cadence. The pipeline wakes once per interval; each plugin's `execution_interval_seconds` decides whether that plugin fires on the tick. By default, the cadence is the gcd of `load_adjustment_interval_seconds` and, when throughput scaling is enabled, `throughput_adjustment_interval_seconds`, preserving existing config fire times. |
| `scheduling.tick_max_duration_seconds` | float | `30.0` | Outer deadline wrapping the full plugin pipeline. Exceeding it aborts the tick; the next tick runs from a clean state. |
| `plugin_registration.transport.request_timeout_seconds` | float | `5.0` | Per-plugin RPC timeout. Plugins exceeding this raise `PluginTimeoutError`; the stage continues with the remaining plugins. |

Existing planner fields still drive the builtin plugins:

- `load_adjustment_interval_seconds` schedules `builtin_load_propose`, which reads FPM and worker-count observations and applies the current load-based algorithm.
- `throughput_adjustment_interval_seconds` schedules `builtin_load_predict` and `builtin_throughput_propose`. The throughput proposer requires the prediction from the same tick, so it only fires when the predict plugin fires.
- When both builtins propose targets in the same tick, load-based scaling runs after throughput-based scaling and preserves the existing behavior: throughput updates the lower-bound replicas, then load-based scaling can adjust above that floor and apply the global GPU budget clamp.
- After the plugin pipeline finishes, the planner applies the same final `min_endpoint` and GPU-budget safety checks to builtin and external-plugin targets before scaling the deployment.

#### DGDR example

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-deployment
spec:
  model: Qwen/Qwen3-0.6B
  features:
    planner:
      optimization_target: sla
      enable_load_scaling: true
      ttft: 200.0
      itl: 10.0
      pre_deployment_sweeping_mode: rapid
      scheduling:
        tick_max_duration_seconds: 30.0
      plugin_registration:
        transport:
          request_timeout_seconds: 5.0
```

## Integration with Profiler

When the profiler runs with planner enabled, it:

1. Selects the best prefill and decode engine configurations
2. Generates engine performance data (prefill TTFT vs ISL, decode ITL vs KV-cache utilization)
3. Saves the `PlannerConfig` and performance data into separate Kubernetes ConfigMaps
4. Adds the planner service to the generated DGD, configured to read from those ConfigMaps

The planner receives its config via `--config /path/to/planner_config.json` which is mounted from the `planner-config-XXXX` ConfigMap. When thorough bootstrap data is generated, profiling data is mounted from the `planner-profile-data-XXXX` ConfigMap.

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
- [Planner Examples](planner-examples.md) — Planner-specific configuration examples
- [DGDR Examples](../../kubernetes/dgdr-examples.md) — DGDR YAML examples, sample configurations, advanced patterns
- [Global Planner Guide](global-planner.md) — Multi-DGD coordination, shared GPU budgets, single-endpoint multi-pool deployments
- [Profiler Guide](../profiler/profiler-guide.md) — How profiling data is generated
