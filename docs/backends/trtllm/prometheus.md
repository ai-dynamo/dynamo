# TensorRT-LLM Prometheus Metrics

**📚 Official Documentation**: [TensorRT-LLM Metrics API Reference](https://nvidia.github.io/TensorRT-LLM/reference/api/tensorrt_llm.metrics.html)

This document describes how TensorRT-LLM Prometheus metrics are exposed in Dynamo.

## Overview

When running TensorRT-LLM through Dynamo, TensorRT-LLM engine metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both TensorRT-LLM engine metrics (prefixed with `trtllm:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

**Note**: TensorRT-LLM does not add a prefix to its metrics by default, but Dynamo automatically adds the `trtllm:` prefix for clarity and consistency with other engines (vLLM, SGLang).

For the complete and authoritative list of all TensorRT-LLM metrics, always refer to the official documentation linked above.

Dynamo runtime metrics are documented in [docs/guides/metrics.md](../../guides/metrics.md).

## Metric Reference

The official documentation includes:
- Complete metric definitions with detailed explanations
- Counter, Gauge, and Histogram metrics
- Metric labels (e.g., `model_name`, `engine_type`) - note that TensorRT-LLM uses `model_name` instead of Dynamo's standard `model` label convention
- Performance and resource usage metrics
- Request lifecycle metrics

## Metric Categories

TensorRT-LLM provides metrics in the following categories (all prefixed with `trtllm:`):
- Request metrics (latency, throughput)
- Performance metrics (TTFT, TPOT, ITL)
- Resource usage (GPU memory, KV cache)
- Scheduler metrics
- Disaggregation metrics (when enabled)

**Note:** Specific metrics are subject to change between TensorRT-LLM versions. Always refer to the [official documentation](https://nvidia.github.io/TensorRT-LLM/reference/api/tensorrt_llm.metrics.html) or inspect the `/metrics` endpoint for your TensorRT-LLM version.

## Enabling Metrics in Dynamo

TensorRT-LLM metrics are automatically exposed when running TensorRT-LLM through Dynamo with the `--publish-events-and-metrics` flag.

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
# HELP trtllm:request_latency_seconds Time between request arrival and completion
# TYPE trtllm:request_latency_seconds histogram
trtllm:request_latency_seconds_bucket{le="0.1",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 5.0
trtllm:request_latency_seconds_bucket{le="0.5",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 25.0
trtllm:request_latency_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:request_latency_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 45.2
# HELP trtllm:time_to_first_token_seconds Time to first token latency
# TYPE trtllm:time_to_first_token_seconds histogram
trtllm:time_to_first_token_seconds_bucket{le="0.01",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 0.0
trtllm:time_to_first_token_seconds_bucket{le="0.05",model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 12.0
trtllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 150.0
trtllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-0.6B",engine_type="trtllm"} 8.75
```

## Implementation Details

- TensorRT-LLM uses the `MetricsCollector` class from `tensorrt_llm.metrics` module
- Metrics are collected when `--publish-events-and-metrics` is enabled
- The integration uses Dynamo's `register_engine_metrics_callback()` function with `add_prefix="trtllm:"`
- Metrics appear after TensorRT-LLM engine initialization completes
- The `MetricsCollector` is initialized with model metadata (model name, engine type)

## Configuration

### Required Flags

To enable metrics collection in TensorRT-LLM through Dynamo:

```bash
python -m dynamo.trtllm --model <model_name> --publish-events-and-metrics
```

### Backend Configuration

The metrics collection is configured in the engine arguments:
- `return_perf_metrics`: Set to `True` when `--publish-events-and-metrics` is enabled
- `backend`: Must be set to `"pytorch"` for metrics collection
- `event_buffer_max_size`: Buffer size for KV cache events (default: 1024)

## See Also

### TensorRT-LLM Metrics
- [Official TensorRT-LLM Metrics API Reference](https://nvidia.github.io/TensorRT-LLM/reference/api/tensorrt_llm.metrics.html)
- [TensorRT-LLM GitHub - Metrics Implementation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/metrics)

### Dynamo Metrics
- **Dynamo Metrics Guide**: See `docs/guides/metrics.md` for complete documentation on Dynamo runtime metrics
- **Dynamo Runtime Metrics**: Metrics prefixed with `dynamo_*` for runtime, components, endpoints, and namespaces
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Available at the same `/metrics` endpoint alongside TensorRT-LLM metrics
- **Integration Code**: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
