---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# TensorRT-LLM Prometheus Metrics

## Overview

When running TensorRT-LLM through Dynamo, TensorRT-LLM's Prometheus metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both TensorRT-LLM engine metrics (prefixed with `trtllm_`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

Additional performance metrics are available via non-Prometheus APIs (see [Non-Prometheus Performance Metrics](#non-prometheus-performance-metrics) below).

As of the date of this documentation, the included TensorRT-LLM version 1.1.0rc5 exposes **5 basic Prometheus metrics**. Note that the `trtllm_` prefix is added by Dynamo.

**For Dynamo runtime metrics**, see the [Dynamo Metrics Guide](../../observability/metrics.md).

**For visualization setup instructions**, see the [Prometheus and Grafana Setup Guide](../../observability/prometheus-grafana.md).

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_SYSTEM_PORT` | System metrics/health port | `-1` (disabled) | `8081` |

## Getting Started Quickly

This is a single machine example.

### Start Observability Stack

For visualizing metrics with Prometheus and Grafana, start the observability stack. See [Observability Getting Started](../../observability/README.md#getting-started-quickly) for instructions.

### Launch Dynamo Components

Launch a frontend and TensorRT-LLM backend to test metrics:

```bash
# Start frontend (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
$ python -m dynamo.frontend

# Enable system metrics server on port 8081 and enable metrics collection
$ DYN_SYSTEM_PORT=8081 python -m dynamo.trtllm --model <model_name> --publish-events-and-metrics
```

**Note:** The `backend` must be set to `"pytorch"` for metrics collection (enforced in `components/src/dynamo/trtllm/main.py`). TensorRT-LLM's `MetricsCollector` integration has only been tested/validated with the PyTorch backend.

Wait for the TensorRT-LLM worker to start, then send requests and check metrics:

```bash
# Send a request
curl -H 'Content-Type: application/json' \
-d '{
  "model": "<model_name>",
  "max_completion_tokens": 100,
  "messages": [{"role": "user", "content": "Hello"}]
}' \
http://localhost:8000/v1/chat/completions

# Check metrics from the worker
curl -s localhost:8081/metrics | grep "^trtllm_"
```

## Exposed Metrics

TensorRT-LLM exposes metrics in Prometheus Exposition Format text at the `/metrics` HTTP endpoint. All TensorRT-LLM engine metrics use the `trtllm_` prefix and include labels (e.g., `model_name`, `engine_type`, `finished_reason`) to identify the source.

**Note:** TensorRT-LLM uses `model_name` instead of Dynamo's standard `model` label convention.

**Example Prometheus Exposition Format text:**

```
# HELP trtllm_request_success_total Count of successfully processed requests.
# TYPE trtllm_request_success_total counter
trtllm_request_success_total{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm",finished_reason="stop"} 150.0
trtllm_request_success_total{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm",finished_reason="length"} 5.0

# HELP trtllm_time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE trtllm_time_to_first_token_seconds histogram
trtllm_time_to_first_token_seconds_bucket{le="0.01",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 0.0
trtllm_time_to_first_token_seconds_bucket{le="0.05",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 12.0
trtllm_time_to_first_token_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm_time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 8.75

# HELP trtllm_e2e_request_latency_seconds Histogram of end to end request latency in seconds.
# TYPE trtllm_e2e_request_latency_seconds histogram
trtllm_e2e_request_latency_seconds_bucket{le="0.5",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 25.0
trtllm_e2e_request_latency_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm_e2e_request_latency_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 45.2

# HELP trtllm_time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE trtllm_time_per_output_token_seconds histogram
trtllm_time_per_output_token_seconds_bucket{le="0.1",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 120.0
trtllm_time_per_output_token_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm_time_per_output_token_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 12.5

# HELP trtllm_request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE trtllm_request_queue_time_seconds histogram
trtllm_request_queue_time_seconds_bucket{le="1.0",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 140.0
trtllm_request_queue_time_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm_request_queue_time_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 32.1
```

**Note:** The specific metrics shown above are examples and may vary depending on your TensorRT-LLM version. Always inspect your actual `/metrics` endpoint for the current list.

### Metric Categories

TensorRT-LLM provides metrics in the following categories (all prefixed with `trtllm_`):

- **Request metrics** - Request success tracking and latency measurements
- **Performance metrics** - Time to first token (TTFT), time per output token (TPOT), and queue time

**Note:** Metrics may change between TensorRT-LLM versions. Always inspect the `/metrics` endpoint for your version.

## Available Metrics

The following metrics are exposed via Dynamo's `/metrics` endpoint (with the `trtllm_` prefix added by Dynamo) for TensorRT-LLM version 1.1.0rc5:

- `trtllm_request_success_total` (Counter) — Count of successfully processed requests by finish reason
  - Labels: `model_name`, `engine_type`, `finished_reason`
- `trtllm_e2e_request_latency_seconds` (Histogram) — End-to-end request latency (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm_time_to_first_token_seconds` (Histogram) — Time to first token, TTFT (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm_time_per_output_token_seconds` (Histogram) — Time per output token, TPOT (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm_request_queue_time_seconds` (Histogram) — Time a request spends waiting in the queue (seconds)
  - Labels: `model_name`, `engine_type`

These metric names and availability are subject to change with TensorRT-LLM version updates.

TensorRT-LLM provides Prometheus metrics through the `MetricsCollector` class (see [tensorrt_llm/metrics/collector.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py)).

## Non-Prometheus Performance Metrics

TensorRT-LLM provides extensive performance data beyond the basic Prometheus metrics. These are not currently exposed to Prometheus.

### Available via Code References

- **RequestPerfMetrics Structure**: [tensorrt_llm/executor/result.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/executor/result.py) - KV cache, timing, speculative decoding metrics
- **Engine Statistics**: `engine.llm.get_stats_async()` - System-wide aggregate statistics
- **KV Cache Events**: `engine.llm.get_kv_cache_events_async()` - Real-time cache operations

### Example RequestPerfMetrics JSON Structure

```json
{
  "timing_metrics": {
    "arrival_time": 1234567890.123,
    "first_scheduled_time": 1234567890.135,
    "first_token_time": 1234567890.150,
    "last_token_time": 1234567890.300,
    "kv_cache_size": 2048576,
    "kv_cache_transfer_start": 1234567890.140,
    "kv_cache_transfer_end": 1234567890.145
  },
  "kv_cache_metrics": {
    "num_total_allocated_blocks": 100,
    "num_new_allocated_blocks": 10,
    "num_reused_blocks": 90,
    "num_missed_blocks": 5
  },
  "speculative_decoding": {
    "acceptance_rate": 0.85,
    "total_accepted_draft_tokens": 42,
    "total_draft_tokens": 50
  }
}
```

**Note:** These structures are valid as of the date of this documentation but are subject to change with TensorRT-LLM version updates.

## Implementation Details

- **Prometheus Integration**: Uses the `MetricsCollector` class from `tensorrt_llm.metrics` (see [collector.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py))
- **Dynamo Integration**: Uses `register_engine_metrics_callback()` function with `add_prefix="trtllm_"`
- **Engine Configuration**: `return_perf_metrics` set to `True` when `--publish-events-and-metrics` is enabled
- **Initialization**: Metrics appear after TensorRT-LLM engine initialization completes
- **Metadata**: `MetricsCollector` initialized with model metadata (model name, engine type)

## Related Documentation

### TensorRT-LLM Metrics
- See the [Non-Prometheus Performance Metrics](#non-prometheus-performance-metrics) section above for detailed performance data and source code references
- [TensorRT-LLM Metrics Collector](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py) - Source code reference

### Dynamo Metrics
- [Dynamo Metrics Guide](../../observability/metrics.md) - Complete documentation on Dynamo runtime metrics
- [Prometheus and Grafana Setup](../../observability/prometheus-grafana.md) - Visualization setup instructions
- Dynamo runtime metrics (prefixed with `dynamo_*`) are available at the same `/metrics` endpoint alongside TensorRT-LLM metrics
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Integration code: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
