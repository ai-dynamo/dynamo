---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner
---

## Why LLM Inference Needs a Different Autoscaler

Scaling a traditional web service is straightforward: watch CPU or request rate, add replicas when load is high, remove them when it's low. Tools like HPA and KEDA work well for this because the relationship between load and latency is roughly linear — twice the requests means roughly twice the CPU, so a simple threshold policy keeps response times stable.

LLM inference breaks these assumptions:

- **Latency depends on request content, not just request count.** A single request with a 32K-token prompt consumes orders of magnitude more compute than a short one. Two requests per second can mean completely different GPU loads depending on input/output sequence lengths.
- **Prefill and decode have different scaling characteristics.** In disaggregated serving, prefill is compute-bound (scales with input length) while decode is memory-bound (scales with concurrent sequences and KV cache usage). A single replica count doesn't capture both.
- **The metrics that matter aren't standard.** The SLAs users care about — Time to First Token (TTFT) and Inter-Token Latency (ITL) — don't map cleanly to CPU utilization or request throughput. HPA can't target "keep P95 TTFT under 500ms" because that requires understanding the relationship between sequence lengths, GPU memory pressure, and latency.
- **Scaling decisions are expensive.** Spinning up a GPU worker takes minutes, not seconds. Overscaling wastes GPU-hours at cloud prices; underscaling violates SLAs. The autoscaler needs to predict demand, not just react to it.

The Dynamo **Planner** is an autoscaler purpose-built for these constraints. It understands engine profiling data, tracks per-worker GPU utilization, predicts traffic patterns, and makes scaling decisions that directly target TTFT and ITL SLAs — not proxy metrics.

> **New to the Planner?** Start with the [Planner Guide](planner-guide.md) for a complete workflow including profiling and deployment.

> **Need multi-DGD coordination?** See the [Global Planner Guide](global-planner.md) for shared-policy coordination across multiple DGDs and single-endpoint multi-pool deployments.

## Scaling Modes

The Planner supports two scaling modes that can run independently or together:

- **Throughput-based scaling**: Uses pre-deployment engine performance data (from self-benchmark or profiler) and traffic prediction to compute the replica count needed to meet TTFT and ITL targets. Adjusts on a longer interval (default 180s). This is the primary mode for production deployments.
- **Load-based scaling**: Uses ForwardPassMetrics (FPM) from the Dynamo event plane and fits an online linear regression to make scaling decisions. No pre-deployment data or KV Router required. Adjusts on a short interval (default 5s) to respond quickly to bursts.

When both modes are enabled, throughput-based scaling provides a capacity floor (long-term planning) while load-based scaling handles real-time adjustments above that floor.

## Feature Matrix

| Feature | Throughput-Based | Load-Based |
|---------|:----------------:|:-------------------------:|
| **Deployment** | | |
| Disaggregated | Supported | Supported |
| Aggregated | Supported | Supported |
| **LLM Framework** | | |
| SGLang | Supported | Supported |
| TensorRT-LLM | Supported | Supported |
| vLLM | Supported | Supported |
| **Requires Pre-deployment Data** | Yes (self-benchmark or profiler) | No |
| **Load Predictors** | ARIMA, Prophet, Kalman, Constant | N/A |
| **Router** | | |
| Any (round-robin, random, etc.) | Supported | Not supported |
| KV Router | Supported | Supported |
| **Connectors** | | |
| KubernetesConnector | Supported | Supported |
| VirtualConnector | Supported | Supported |

## When to Use Which Mode

- **Throughput-based scaling** should be enabled whenever engine performance data is available (through self-benchmark or pre-deployment profiling). It provides stable, prediction-based capacity planning.
- **Load-based scaling** should be enabled when traffic is bursty or hard to predict. It reacts quickly to real-time load changes without requiring pre-deployment data.
- **Both modes together**: For the best of both worlds, enable both. Throughput-based scaling provides a lower bound (long-term capacity), while load-based scaling handles bursts above that floor. When both are enabled, use a longer `--adjustment-interval` for throughput-based scaling.

## Quick Start

### Prerequisites

- Dynamo platform installed on Kubernetes ([Installation Guide](../../kubernetes/installation-guide.md))
- kube-prometheus-stack installed ([Metrics Setup](../../kubernetes/observability/metrics.md))

For throughput-based scaling, pre-deployment engine performance data is also required (via self-benchmark mode or [Profiling Guide](../profiler/profiler-guide.md)).

### Throughput-Based Scaling (with DGDR)

The fastest path to a throughput-based planner deployment is through a DynamoGraphDeploymentRequest, which automatically profiles your model:

```bash
kubectl apply -f components/src/dynamo/profiler/deploy/profile_sla_aic_dgdr.yaml -n $NAMESPACE
```

See [Planner Guide](planner-guide.md) for the full workflow.

### Load-Based Scaling (without profiling)

To deploy with load-based scaling only (no profiling required), add these arguments to the planner service in your DGD:

```yaml
args:
  - --enable-loadbased-scaling
  - --disable-throughput-scaling
  - --loadbased-adjustment-interval=5
```

The planner will auto-discover the frontend metrics endpoint from the DGD. See [disagg_planner.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/disagg_planner.yaml) for a complete example.

### Manual DGD Deployment

For manual control with throughput-based scaling, use the disaggregated planner templates:

```bash
# After profiling is complete
kubectl apply -f examples/backends/vllm/deploy/disagg_planner.yaml -n $NAMESPACE
```

## Current Limitations

### Load-based scaling

Load-based scaling has the following known limitations. Throughput-based scaling is not affected by any of these.

**Requires ForwardPassMetrics (FPM).** Load-based scaling uses per-engine per-iteration metrics delivered via the Dynamo event plane (ForwardPassMetrics). FPM is currently only available for vllm and is automatically enabled when the engine uses `InstrumentedScheduler` and `DYN_FORWARDPASS_METRIC_PORT` is set. The KV Router is **not** required for load-based scaling.

### General

**In-flight requests during scale-down.** When the Planner scales down a worker, the worker is terminated without waiting for in-flight requests to complete. Requests that were mid-prefill on the terminated worker will fail. In disaggregated deployments, this can also affect decode workers that were waiting on KV cache transfers from the terminated prefill worker. **Workaround:** Set `--min-endpoint` to a value that avoids scaling below your steady-state traffic floor, and use a lower `--loadbased-scaling-down-sensitivity` value to reduce the frequency of scale-down events.

## Documentation

| Document | Description |
|----------|-------------|
| [Planner Guide](planner-guide.md) | Deployment, configuration, integration |
| [Planner Design](../../design-docs/planner-design.md) | Architecture and algorithm internals |
| [Planner Examples](planner-examples.md) | DGDR YAML examples, sample configurations, advanced patterns |
| [Global Planner Guide](global-planner.md) | Multi-DGD coordination, shared GPU budgets, single-endpoint multi-pool deployments |

## Configuration Reference

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| **Common** | | |
| `--namespace` | `$DYN_NAMESPACE` or `dynamo` | Dynamo logical namespace |
| `--backend` | `vllm` | Backend framework (`sglang`, `trtllm`, `vllm`) |
| `--mode` | `disagg` | Planner mode (`disagg`, `prefill`, `decode`, `agg`) |
| `--environment` | `kubernetes` | Deployment environment |
| `--ttft` | `500.0` | Target Time To First Token (ms) |
| `--itl` | `50.0` | Target Inter-Token Latency (ms) |
| `--max-gpu-budget` | `8` | Maximum GPUs across all workers |
| `--min-endpoint` | `1` | Minimum replicas per worker type |
| `--decode-engine-num-gpu` | `1` | GPUs per decode engine |
| `--prefill-engine-num-gpu` | `1` | GPUs per prefill engine |
| `--no-operation` | `false` | Observation mode (no actual scaling) |
| **Throughput-based scaling** | | |
| `--enable-throughput-scaling` | `true` | Enable throughput-based scaling |
| `--adjustment-interval` | `180` | Seconds between throughput-based scaling decisions |
| `--profile-results-dir` | `profiling_results` | Path to profiling data (NPZ/JSON) |
| `--load-predictor` | `arima` | Prediction model (`arima`, `prophet`, `kalman`, `constant`) |
| **Load-based scaling** | | |
| `--enable-loadbased-scaling` | `false` | Enable load-based scaling |
| `--loadbased-adjustment-interval` | `5` | Seconds between FPM regression updates and load-based scaling decisions |
| `--max-num-fpm-samples` | `64` | Maximum retained FPM observations for regression |
| `--fpm-sample-bucket-size` | `16` | Number of buckets for observation retirement (must be perfect square) |
| `--loadbased-scaling-down-sensitivity` | `80` | Scale-down sensitivity 0-100 (0=never, 100=aggressive) |
| `--loadbased-metric-samples` | `10` | Number of metric samples per adjustment interval |
| `--loadbased-min-observations` | `5` | Minimum observations before regression activates |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_NAMESPACE` | `dynamo` | Dynamo logical namespace |
| `DYN_PARENT_DGD_K8S_NAME` | (required) | Parent DGD K8s resource name |
| `PROMETHEUS_ENDPOINT` | `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090` | Prometheus URL |
| `PLANNER_PROMETHEUS_PORT` | `0` (disabled) | Port for planner's own Prometheus metrics |

## Monitoring

### Grafana Dashboard

Deploy the planner dashboard:

```bash
kubectl apply -n monitoring -f deploy/observability/k8s/grafana-planner-dashboard-configmap.yaml
```

The dashboard shows:
- Worker counts and GPU usage over time
- Observed TTFT, ITL, request rate, sequence lengths
- Predicted load and recommended replica counts
- FPM regression model status

### Prometheus Metrics

When `PLANNER_PROMETHEUS_PORT` is set, the planner serves its own metrics endpoint. Exported series use the `dynamo_planner_*` naming convention (underscores and standard unit suffixes), replacing older `planner:*`-style names.

**Throughput-based scaling** pulls traffic metrics from the cluster-wide Prometheus server:
- Request count and duration
- TTFT and ITL distributions
- Input/output sequence lengths

**Load-based scaling** uses ForwardPassMetrics (FPM) from the Dynamo event plane:
- Per-iteration wall time, scheduled prefill/decode tokens, and queued request status
- Delivered via `FpmEventSubscriber` with automatic engine discovery and lifecycle tracking
- No router `/metrics` scraping required

Core gauges on the planner port include replica counts (`dynamo_planner_num_prefill_replicas`, `dynamo_planner_num_decode_replicas`), observed traffic (`dynamo_planner_observed_*`), replica decisions (`dynamo_planner_predicted_num_prefill_replicas`, `dynamo_planner_predicted_num_decode_replicas`), and cumulative `dynamo_planner_gpu_hours`.

Throughput prediction gauges `dynamo_planner_predicted_requests_per_second`, `dynamo_planner_predicted_input_sequence_tokens`, and `dynamo_planner_predicted_output_sequence_tokens` are wired from throughput-scaling traffic prediction and exposed alongside observed sequence-length metrics.

#### Diagnostics metrics

Additional series support dashboards and offline analysis:

- **Regression-based latency estimates:** `dynamo_planner_estimated_ttft_ms` and `dynamo_planner_estimated_itl_ms` reflect the maximum estimated TTFT and ITL from the online regression across engines.
- **Engine capacity:** `dynamo_planner_engine_prefill_requests_per_second` and `dynamo_planner_engine_decode_requests_per_second` report single-engine prefill and decode capacity under the configured SLA.
- **Scaling decision reasons:** `dynamo_planner_load_scaling_decision` and `dynamo_planner_throughput_scaling_decision` are Enum gauges whose state labels encode why each mode chose to scale, hold, or skip (for example `scale_up`, `no_fpm_data`, `set_lower_bound`).
- **Per-engine FPM queue depths:** `dynamo_planner_engine_queued_prefill_tokens`, `dynamo_planner_engine_queued_decode_kv_tokens`, and `dynamo_planner_engine_inflight_decode_kv_tokens` are labeled with `worker_id` and `dp_rank` for each engine.

### HTML diagnostics reports

The planner can emit periodic, self-contained HTML diagnostics files with interactive Plotly charts.

Configure this in `PlannerConfig` (or the equivalent YAML / constructor wiring your deployment uses):

- `report_interval_hours`: interval in **simulated** time between reports (default `24.0` hours); set to `None` to disable.
- `report_output_dir`: directory where HTML files are written (default `./planner_reports`).
- `live_dashboard_port`: port for a real-time HTTP dashboard (default `0` = disabled). When set, an aiohttp server starts on the given port and serves the current accumulated snapshot data as an interactive Plotly report at `http://<host>:<port>/`. Unlike periodic reports, the live dashboard does **not** clear snapshots — it always shows all data accumulated since the last periodic report (or since startup if periodic reports are disabled).

Reports aggregate per-tick snapshots and use `TickInput.now_s` for timestamps, so they behave the same in live runs (wall clock) and in **replay** with a simulated clock. Typical charts cover worker counts, observed versus estimated latencies versus SLA targets, request rate, engine capacity, scaling decision timelines, and input/output sequence lengths.
