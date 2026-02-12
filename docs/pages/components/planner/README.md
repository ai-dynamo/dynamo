---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Planner

The Planner monitors system performance and automatically scales prefill/decode workers to meet latency SLAs. It runs as a component inside the Dynamo inference graph on Kubernetes.

> **New to the Planner?** Start with the [SLA Planner Quick Start Guide](planner-guide.md) for a complete workflow including profiling and deployment.

## Feature Matrix

| Category | Feature | Status |
|----------|---------|--------|
| **Backend** | Local (bare metal) | Deprecated |
| | Kubernetes | Supported |
| **LLM Framework** | vLLM | Supported |
| | TensorRT-LLM | Supported |
| | SGLang | Supported |
| **Serving Type** | Aggregated | Unsupported |
| | Disaggregated | Supported |
| **Scaling Mode** | SLA-based (TTFT/ITL targets) | Supported (primary) |
| | Load-based (KV cache/queue thresholds) | Deprecated |
| **Load Predictors** | ARIMA | Supported |
| | Prophet | Supported |
| | Kalman filter | Supported |
| | Constant (current = next) | Supported |
| **Connectors** | KubernetesConnector (native DGD scaling) | Supported |
| | VirtualConnector (external environments) | Supported |

## Quick Start

### Prerequisites

- Dynamo platform installed on Kubernetes ([Installation Guide](../../kubernetes/installation-guide.md))
- kube-prometheus-stack installed ([Metrics Setup](../../kubernetes/observability/metrics.md))
- Pre-deployment profiling completed ([Profiling Guide](../profiler/profiler-guide.md))

### Deploy with DGDR (Recommended)

The fastest path to a planner-enabled deployment is through a DynamoGraphDeploymentRequest:

```bash
kubectl apply -f benchmarks/profiler/deploy/profile_sla_aic_dgdr.yaml -n $NAMESPACE
```

This automatically profiles your model and deploys with the SLA planner. See [SLA Planner Guide](planner-guide.md) for the full workflow.

### Deploy with DGD (Manual)

For manual control, use the disaggregated planner templates:

```bash
# After profiling is complete
kubectl apply -f examples/backends/vllm/deploy/disagg_planner.yaml -n $NAMESPACE
```

## Documentation

| Document | Description |
|----------|-------------|
| [Planner Guide](planner-guide.md) | Deployment, configuration, integration, troubleshooting |
| [Planner Examples](planner-examples.md) | DGDR YAML examples, sample configurations, advanced patterns |
| [SLA Planner Guide](planner-guide.md) | End-to-end DGDR workflow: define SLAs, profile, deploy, monitor |
| [SLA-based Planner](planner-guide.md) | Scaling algorithm, correction factors, load prediction details |
| [Load-based Planner](README.md) | Legacy load-based scaling (deprecated) |
| [SLA-Driven Profiling](../profiler/profiler-guide.md) | Pre-deployment profiling process and configuration |
| [Planner Design](../../design-docs/planner-design.md) | Architecture deep-dive for contributors |

## Configuration Reference

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--namespace` | `$DYN_NAMESPACE` or `dynamo` | Dynamo logical namespace |
| `--backend` | `vllm` | Backend framework (`vllm`, `sglang`, `trtllm`) |
| `--environment` | `kubernetes` | Deployment environment |
| `--adjustment-interval` | `180` | Seconds between scaling decisions |
| `--ttft` | `500.0` | Target Time To First Token (ms) |
| `--itl` | `50.0` | Target Inter-Token Latency (ms) |
| `--isl` | `3000` | Expected average input sequence length |
| `--osl` | `150` | Expected average output sequence length |
| `--load-predictor` | `arima` | Prediction model (`arima`, `prophet`, `kalman`, `constant`) |
| `--max-gpu-budget` | `8` | Maximum GPUs across all workers |
| `--min-endpoint` | `1` | Minimum replicas per worker type |
| `--decode-engine-num-gpu` | `1` | GPUs per decode engine |
| `--prefill-engine-num-gpu` | `1` | GPUs per prefill engine |
| `--no-operation` | `false` | Observation mode (no actual scaling) |
| `--no-correction` | `false` | Disable correction factors |
| `--profile-results-dir` | `profiling_results` | Path to profiling data (NPZ/JSON) |

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

The planner queries the frontend's `/metrics` endpoint via Prometheus. Required metrics:
- Request count and duration
- TTFT and ITL distributions
- Input/output sequence lengths
