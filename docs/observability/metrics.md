<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Metrics

## Overview

Dynamo provides built-in metrics capabilities through the Dynamo metrics API, which is automatically available whenever you use the `DistributedRuntime` framework. This document serves as a reference for all available metrics in Dynamo.

**For visualization setup instructions**, see the [Prometheus and Grafana Setup Guide](prometheus-grafana.md).

**For creating custom metrics**, see the [Metrics Developer Guide](metrics-developer-guide.md).

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_SYSTEM_PORT` | Backend component metrics/health port | `-1` (disabled) | `8081` |
| `DYN_HTTP_PORT` | Frontend HTTP port (also configurable via `--http-port` flag) | `8000` | `8000` |

## Getting Started Quickly

This is a single machine example.

### Start Observability Stack

For visualizing metrics with Prometheus and Grafana, start the observability stack. See [Observability Getting Started](README.md#getting-started-quickly) for instructions.


### Launch Dynamo Components

Launch a frontend and vLLM backend to test metrics:

```bash
# Start frontend (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
$ python -m dynamo.frontend

# Enable backend worker's system metrics on port 8081
$ DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B  \
   --enforce-eager --no-enable-prefix-caching --max-num-seqs 3
```

Wait for the vLLM worker to start, then send requests and check metrics:

```bash
# Send a request
curl -H 'Content-Type: application/json' \
-d '{
  "model": "Qwen/Qwen3-0.6B",
  "max_completion_tokens": 100,
  "messages": [{"role": "user", "content": "Hello"}]
}' \
http://localhost:8000/v1/chat/completions

# Check metrics from the backend worker
curl -s localhost:8081/metrics | grep dynamo_component
```

## Exposed Metrics

Dynamo exposes metrics in Prometheus Exposition Format text at the `/metrics` HTTP endpoint. All Dynamo-generated metrics use the `dynamo_*` prefix and include labels (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`) to identify the source component.

**Example Prometheus Exposition Format text:**

```
# HELP dynamo_component_requests_total Total requests processed
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 42

# HELP dynamo_component_request_duration_seconds Request processing time
# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_bucket{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate",le="0.005"} 10
dynamo_component_request_duration_seconds_bucket{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate",le="0.01"} 15
dynamo_component_request_duration_seconds_bucket{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate",le="+Inf"} 42
dynamo_component_request_duration_seconds_sum{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 2.5
dynamo_component_request_duration_seconds_count{dynamo_namespace="default",dynamo_component="worker",dynamo_endpoint="generate"} 42
```

### Metric Categories

Dynamo exposes several categories of metrics:

- **Frontend Metrics** (`dynamo_frontend_*`) - Request handling, token processing, and latency measurements
- **Component Metrics** (`dynamo_component_*`) - Request counts, processing times, byte transfers, and system uptime
- **Specialized Component Metrics** (e.g., `dynamo_preprocessor_*`) - Component-specific metrics
- **Engine Metrics** (Pass-through) - Backend engines expose their own metrics: [vLLM](../backends/vllm/prometheus.md) (`vllm:*`), [SGLang](../backends/sglang/prometheus.md) (`sglang:*`), [TensorRT-LLM](../backends/trtllm/prometheus.md) (`trtllm_*`)

## Runtime Hierarchy

The Dynamo metrics API is available on `DistributedRuntime`, `Namespace`, `Component`, and `Endpoint`, providing a hierarchical approach to metric collection that matches Dynamo's distributed architecture:

- `DistributedRuntime`: Global metrics across the entire runtime
- `Namespace`: Metrics scoped to a specific dynamo_namespace
- `Component`: Metrics for a specific dynamo_component within a namespace
- `Endpoint`: Metrics for individual dynamo_endpoint within a component

This hierarchical structure allows you to create metrics at the appropriate level of granularity for your monitoring needs.

## Available Metrics

### Backend Component Metrics

**Backend workers** (`python -m dynamo.vllm`, `python -m dynamo.sglang`, etc.) expose `dynamo_component_*` metrics on port 8081 by default (configurable via `DYN_SYSTEM_PORT`).

The core Dynamo backend system automatically exposes metrics on the system status port (default: 8081, configurable via `DYN_SYSTEM_PORT`) at the `/metrics` endpoint with the `dynamo_component_*` prefix for all components that use the `DistributedRuntime` framework:

- `dynamo_component_inflight_requests`: Requests currently being processed (gauge)
- `dynamo_component_request_bytes_total`: Total bytes received in requests (counter)
- `dynamo_component_request_duration_seconds`: Request processing time (histogram)
- `dynamo_component_requests_total`: Total requests processed (counter)
- `dynamo_component_response_bytes_total`: Total bytes sent in responses (counter)
- `dynamo_component_uptime_seconds`: DistributedRuntime uptime (gauge)

**Access backend component metrics:**
```bash
# Default port 8081
curl http://localhost:8081/metrics

# Or with custom port
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model <model>
curl http://localhost:8081/metrics
```

### KV Router Statistics (kvstats)

KV router statistics are automatically exposed by LLM workers and KV router components on the backend system status port (port 8081) with the `dynamo_component_kvstats_*` prefix. These metrics provide insights into GPU memory usage and cache efficiency:

- `dynamo_component_kvstats_active_blocks`: Number of active KV cache blocks currently in use (gauge)
- `dynamo_component_kvstats_total_blocks`: Total number of KV cache blocks available (gauge)
- `dynamo_component_kvstats_gpu_cache_usage_percent`: GPU cache usage as a percentage (0.0-1.0) (gauge)
- `dynamo_component_kvstats_gpu_prefix_cache_hit_rate`: GPU prefix cache hit rate as a percentage (0.0-1.0) (gauge)

These metrics are published by:
- **LLM Workers**: vLLM and TRT-LLM backends publish these metrics through their respective publishers
- **KV Router**: The KV router component aggregates and exposes these metrics for load balancing decisions

### Specialized Component Metrics

Some components expose additional metrics specific to their functionality:

- `dynamo_preprocessor_*`: Metrics specific to preprocessor components

### Frontend Metrics

**Important:** The frontend and backend workers are separate components that expose metrics on different ports. See [Backend Component Metrics](#backend-component-metrics) for backend metrics.

The Dynamo HTTP Frontend (`python -m dynamo.frontend`) exposes `dynamo_frontend_*` metrics on port 8000 by default (configurable via `--http-port` or `DYN_HTTP_PORT`) at the `/metrics` endpoint. Most metrics include `model` labels containing the model name:

- `dynamo_frontend_inflight_requests`: Inflight requests (gauge)
- `dynamo_frontend_queued_requests`: Number of requests in HTTP processing queue (gauge)
- `dynamo_frontend_disconnected_clients`: Number of disconnected clients (gauge)
- `dynamo_frontend_input_sequence_tokens`: Input sequence length (histogram)
- `dynamo_frontend_cached_tokens`: Number of cached tokens (prefix cache hits) per request (histogram)
- `dynamo_frontend_inter_token_latency_seconds`: Inter-token latency (histogram)
- `dynamo_frontend_output_sequence_tokens`: Output sequence length (histogram)
- `dynamo_frontend_output_tokens_total`: Total number of output tokens generated (counter)
- `dynamo_frontend_request_duration_seconds`: LLM request duration (histogram)
- `dynamo_frontend_requests_total`: Total LLM requests (counter)
- `dynamo_frontend_time_to_first_token_seconds`: Time to first token (histogram)

**Access frontend metrics:**
```bash
curl http://localhost:8000/metrics
```

**Note**: The `dynamo_frontend_inflight_requests` metric tracks requests from HTTP handler start until the complete response is finished, while `dynamo_frontend_queued_requests` tracks requests from HTTP handler start until first token generation begins (including prefill time). HTTP queue time is a subset of inflight time.

#### Model Configuration Metrics

The frontend also exposes model configuration metrics (on port 8000 `/metrics` endpoint) with the `dynamo_frontend_model_*` prefix. These metrics are populated from the worker backend registration service when workers register with the system. All model configuration metrics include a `model` label.

**Runtime Config Metrics (from ModelRuntimeConfig):**
These metrics come from the runtime configuration provided by worker backends during registration.

- `dynamo_frontend_model_total_kv_blocks`: Total KV blocks available for a worker serving the model (gauge)
- `dynamo_frontend_model_max_num_seqs`: Maximum number of sequences for a worker serving the model (gauge)
- `dynamo_frontend_model_max_num_batched_tokens`: Maximum number of batched tokens for a worker serving the model (gauge)

**MDC Metrics (from ModelDeploymentCard):**
These metrics come from the Model Deployment Card information provided by worker backends during registration. Note that when multiple worker instances register with the same model name, only the first instance's configuration metrics (runtime config and MDC metrics) will be populated. Subsequent instances with duplicate model names will be skipped for configuration metric updates.

- `dynamo_frontend_model_context_length`: Maximum context length for a worker serving the model (gauge)
- `dynamo_frontend_model_kv_cache_block_size`: KV cache block size for a worker serving the model (gauge)
- `dynamo_frontend_model_migration_limit`: Request migration limit for a worker serving the model (gauge)

### Request Processing Flow

This section explains the distinction between two key metrics used to track request processing:

1. **Inflight**: Tracks requests from HTTP handler start until the complete response is finished
2. **HTTP Queue**: Tracks requests from HTTP handler start until first token generation begins (including prefill time)

**Example Request Flow:**
```
curl -s localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Hello let's talk about LLMs",
  "stream": false,
  "max_tokens": 1000
}'
```

**Timeline:**
```
Timeline:    0, 1, ...
Client ────> Frontend:8000 ────────────────────> Dynamo component/backend (vLLM, SGLang, TRT)
             │request start                     │received                              │
             |                                  |                                      |
             │                                  ├──> start prefill ──> first token ──> |last token
             │                                  │     (not impl)       |               |
             ├─────actual HTTP queue¹ ──────────┘                      │               |
             │                                                         │               │
             ├─────implemented HTTP queue ─────────────────────────────┘               |
             │                                                                         │
             └─────────────────────────────────── Inflight ────────────────────────────┘
```

**Concurrency Example:**
Suppose the backend allows 3 concurrent requests and there are 10 clients continuously hitting the frontend:
- All 10 requests will be counted as inflight (from start until complete response)
- 7 requests will be in HTTP queue most of the time
- 3 requests will be actively processed (between first token and last token)

**Key Differences:**
- **Inflight**: Measures total request lifetime including processing time
- **HTTP Queue**: Measures queuing time before processing begins (including prefill time)
- **HTTP Queue ≤ Inflight** (HTTP queue is a subset of inflight time)

## Related Documentation

- [Distributed Runtime Architecture](../design_docs/distributed_runtime.md)
- [Dynamo Architecture Overview](../design_docs/architecture.md)
- [Backend Guide](../development/backend-guide.md)
