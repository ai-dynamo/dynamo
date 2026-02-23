---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner
---

# Planner

The Planner monitors system performance and automatically scales prefill/decode workers to meet latency SLAs. It runs as a component inside the Dynamo inference graph on Kubernetes.

The SLA Planner supports two scaling modes:

- **Throughput-based scaling**: Uses pre-deployment profiling data and traffic prediction to compute the number of replicas needed to meet TTFT and ITL SLA targets. This is the primary scaling mode for production deployments.
- **Load-based scaling (Experimental)**: Uses real-time per-worker load metrics (active prefill tokens, active KV blocks) from the router to make SLA-aware scaling decisions via online linear regression. Does not require profiling data. Responds quickly to traffic bursts.

When both modes are enabled, throughput-based scaling provides a lower bound on replicas (long-term capacity planning) while load-based scaling handles real-time adjustments (burst response).

> **New to the Planner?** Start with the [SLA Planner Quick Start Guide](planner-guide.md) for a complete workflow including profiling and deployment.

## Feature Matrix

| Feature | Throughput-Based | Load-Based (Experimental) |
|---------|:----------------:|:-------------------------:|
| **Deployment** | | |
| Disaggregated | Supported | Supported |
| Aggregated | Unsupported | Supported |
| **LLM Framework** | | |
| vLLM | Supported | Supported |
| TensorRT-LLM | Supported | Supported |
| SGLang | Supported | Supported |
| **Requires Profiling Data** | Yes | No |
| **Load Predictors** | ARIMA, Prophet, Kalman, Constant | N/A |
| **Connectors** | | |
| KubernetesConnector | Supported | Supported |
| VirtualConnector | Supported | Supported |

## When to Use Which Mode

- **Throughput-based scaling** should be enabled whenever engine profiling data is available (through pre-deployment profiling). It provides stable, prediction-based capacity planning.
- **Load-based scaling** should be enabled when traffic is bursty or hard to predict. It reacts quickly to real-time load changes without requiring profiling data.
- **Both modes together**: For the best of both worlds, enable both. Throughput-based scaling provides a lower bound (long-term capacity), while load-based scaling handles bursts above that floor. When both are enabled, use a longer `--adjustment-interval` for throughput-based scaling.

## Quick Start

### Prerequisites

- Dynamo platform installed on Kubernetes ([Installation Guide](../../kubernetes/installation-guide.md))
- kube-prometheus-stack installed ([Metrics Setup](../../kubernetes/observability/metrics.md))

For throughput-based scaling, pre-deployment profiling is also required ([Profiling Guide](../profiler/profiler-guide.md)).

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

The planner will auto-discover the frontend metrics endpoint from the DGD. See [disagg_planner_load.yaml](../../../../tests/planner/scaling/disagg_planner_load.yaml) for a complete example.

### Manual DGD Deployment

For manual control with throughput-based scaling, use the disaggregated planner templates:

```bash
# After profiling is complete
kubectl apply -f examples/backends/vllm/deploy/disagg_planner.yaml -n $NAMESPACE
```

## Documentation

| Document | Description |
|----------|-------------|
| [Planner Guide](planner-guide.md) | Deployment, configuration, integration, troubleshooting |
| [Planner Examples](planner-examples.md) | DGDR YAML examples, sample configurations, advanced patterns |
| [SLA-Driven Profiling](../profiler/profiler-guide.md) | Pre-deployment profiling process and configuration |
| [Planner Design](../../design-docs/planner-design.md) | Architecture deep-dive for contributors |

## Configuration Reference

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| **Common** | | |
| `--namespace` | `$DYN_NAMESPACE` or `dynamo` | Dynamo logical namespace |
| `--backend` | `vllm` | Backend framework (`vllm`, `sglang`, `trtllm`) |
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
| `--no-correction` | `false` | Disable correction factors |
| **Load-based scaling (Experimental)** | | |
| `--enable-loadbased-scaling` | `false` | Enable load-based scaling |
| `--disable-throughput-scaling` | `false` | Disable throughput-based scaling (required for `agg` mode) |
| `--loadbased-router-metrics-url` | auto-discovered | URL to router's `/metrics` endpoint |
| `--loadbased-adjustment-interval` | `5` | Seconds between load-based scaling decisions |
| `--loadbased-learning-window` | `50` | Sliding window size for regression model |
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
- Correction factors (actual vs. expected performance)

### Prometheus Metrics

**Throughput-based scaling** pulls traffic metrics from the cluster-wide Prometheus server:
- Request count and duration
- TTFT and ITL distributions
- Input/output sequence lengths

**Load-based scaling** pulls per-engine status directly from the frontend's `/metrics` endpoint:
- Active prefill tokens per worker
- Active decode blocks per worker
- Last observed TTFT, ITL, and ISL per worker
