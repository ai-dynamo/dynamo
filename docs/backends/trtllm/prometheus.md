# TensorRT-LLM Prometheus Metrics

This document describes how TensorRT-LLM Prometheus metrics are exposed in Dynamo, as well as where to find non-Prometheus metrics.

## Overview

When running TensorRT-LLM through Dynamo, TensorRT-LLM's Prometheus metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both TensorRT-LLM engine metrics (prefixed with `trtllm:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

Additional performance metrics are available via non-Prometheus APIs in the RequestPerfMetrics section below.

As of the date of this documentation, the included TensorRT-LLM version 1.1.0rc5 exposes **5 basic Prometheus metrics**. Note that the `trtllm:` prefix is added by Dynamo.

Dynamo runtime metrics are documented in [docs/guides/metrics.md](../../guides/metrics.md).

## Metric Reference

TensorRT-LLM provides Prometheus metrics through the `MetricsCollector` class (see [tensorrt_llm/metrics/collector.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py)), which includes:
- Counter and Histogram metrics
- Metric labels (e.g., `model_name`, `engine_type`, `finished_reason`) - note that TensorRT-LLM uses `model_name` instead of Dynamo's standard `model` label convention

### Current Prometheus Metrics (TensorRT-LLM 1.1.0rc5)

The following metrics are exposed via Dynamo's `/metrics` endpoint (with the `trtllm:` prefix added by Dynamo):

- `trtllm:request_success_total` (Counter) — Count of successfully processed requests by finish reason
  - Labels: `model_name`, `engine_type`, `finished_reason`
- `trtllm:e2e_request_latency_seconds` (Histogram) — End-to-end request latency (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm:time_to_first_token_seconds` (Histogram) — Time to first token, TTFT (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm:time_per_output_token_seconds` (Histogram) — Time per output token, TPOT (seconds)
  - Labels: `model_name`, `engine_type`
- `trtllm:request_queue_time_seconds` (Histogram) — Time a request spends waiting in the queue (seconds)
  - Labels: `model_name`, `engine_type`

These metric names and availability are subject to change with TensorRT-LLM version updates.

## Metric Categories

TensorRT-LLM provides metrics in the following categories (all prefixed with `trtllm:`):
- Request metrics (latency, throughput)
- Performance metrics (TTFT, TPOT, queue time)

**Note:** Metrics may change between TensorRT-LLM versions. Always inspect the `/metrics` endpoint for your version.

## Enabling Metrics in Dynamo

TensorRT-LLM Prometheus metrics are automatically exposed when running TensorRT-LLM through Dynamo with the `--publish-events-and-metrics` flag.

### Required Configuration
```bash
python -m dynamo.trtllm --model <model_name> --publish-events-and-metrics
```

### Backend Requirement
- `backend`: Must be set to `"pytorch"` for metrics collection (enforced in `components/src/dynamo/trtllm/main.py`)
- TensorRT-LLM's `MetricsCollector` integration has only been tested/validated with the PyTorch backend

## Inspecting Metrics

To see the actual metrics available in your TensorRT-LLM version:

### 1. Launch TensorRT-LLM with Metrics Enabled

```bash
# Set environment variables
export DYN_SYSTEM_ENABLED=true
export DYN_SYSTEM_PORT=8081

# Start TensorRT-LLM worker with metrics enabled
python -m dynamo.trtllm --model <model_name> --publish-events-and-metrics

# Wait for engine to initialize
```

Metrics will be available at: `http://localhost:8081/metrics`

### 2. Fetch Metrics via curl

```bash
curl http://localhost:8081/metrics | grep "^trtllm:"
```

### 3. Example Output

**Note:** The specific metrics shown below are examples and may vary depending on your TensorRT-LLM version. Always inspect your actual `/metrics` endpoint for the current list.

```
# HELP trtllm:request_success_total Count of successfully processed requests.
# TYPE trtllm:request_success_total counter
trtllm:request_success_total{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm",finished_reason="stop"} 150.0
trtllm:request_success_total{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm",finished_reason="length"} 5.0

# HELP trtllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE trtllm:time_to_first_token_seconds histogram
trtllm:time_to_first_token_seconds_bucket{le="0.01",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 0.0
trtllm:time_to_first_token_seconds_bucket{le="0.05",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 12.0
trtllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 8.75

# HELP trtllm:e2e_request_latency_seconds Histogram of end to end request latency in seconds.
# TYPE trtllm:e2e_request_latency_seconds histogram
trtllm:e2e_request_latency_seconds_bucket{le="0.5",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 25.0
trtllm:e2e_request_latency_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:e2e_request_latency_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 45.2

# HELP trtllm:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE trtllm:time_per_output_token_seconds histogram
trtllm:time_per_output_token_seconds_bucket{le="0.1",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 120.0
trtllm:time_per_output_token_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:time_per_output_token_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 12.5

# HELP trtllm:request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE trtllm:request_queue_time_seconds histogram
trtllm:request_queue_time_seconds_bucket{le="1.0",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 140.0
trtllm:request_queue_time_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:request_queue_time_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 32.1
```

## Implementation Details

- **Prometheus Integration**: Uses the `MetricsCollector` class from `tensorrt_llm.metrics` (see [collector.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py))
- **Dynamo Integration**: Uses `register_engine_metrics_callback()` function with `add_prefix="trtllm:"`
- **Engine Configuration**: `return_perf_metrics` set to `True` when `--publish-events-and-metrics` is enabled
- **Initialization**: Metrics appear after TensorRT-LLM engine initialization completes
- **Metadata**: `MetricsCollector` initialized with model metadata (model name, engine type)

## TensorRT-LLM Specific: Non-Prometheus Performance Metrics

TensorRT-LLM provides extensive performance data beyond the basic Prometheus metrics. These are **not exposed to Prometheus**.

### Available via Code References:
- **RequestPerfMetrics Structure**: [tensorrt_llm/executor/result.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/executor/result.py) - KV cache, timing, speculative decoding metrics
- **Engine Statistics**: `engine.llm.get_stats_async()` - System-wide aggregate statistics
- **KV Cache Events**: `engine.llm.get_kv_cache_events_async()` - Real-time cache operations

### Example RequestPerfMetrics JSON Structure:
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

**Note**: These structures are valid as of the date of this documentation but are subject to change with TensorRT-LLM version updates.

## See Also

### TensorRT-LLM Metrics
- See the "TensorRT-LLM Specific: Non-Prometheus Performance Metrics" section above for detailed performance data and source code references

### Dynamo Metrics
- **Dynamo Metrics Guide**: See [docs/guides/metrics.md](../../guides/metrics.md) for complete documentation on Dynamo runtime metrics
- **Dynamo Runtime Metrics**: Metrics prefixed with `dynamo_*` for runtime, components, endpoints, and namespaces
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Available at the same `/metrics` endpoint alongside TensorRT-LLM metrics
- **Integration Code**: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
