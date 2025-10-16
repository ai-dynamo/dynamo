# vLLM Prometheus Metrics

**📚 Official Documentation**: [vLLM Metrics Design](https://docs.vllm.ai/en/latest/design/metrics.html)

This document describes how vLLM Prometheus metrics are exposed in Dynamo.

## Overview

When running vLLM through Dynamo, vLLM engine metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both vLLM engine metrics (prefixed with `vllm:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

For the complete and authoritative list of all vLLM metrics, always refer to the official documentation linked above.

Dynamo runtime metrics are documented in [docs/guides/metrics.md](../../guides/metrics.md).

## Metric Reference

The official documentation includes:
- Complete metric definitions with detailed explanations
- Counter, Gauge, and Histogram metrics
- Metric labels (e.g., `model_name`, `finished_reason`, `scheduling_event`)
- Design rationale and implementation details
- Information about v1 metrics migration
- Future work and deprecated metrics

## Metric Categories

vLLM provides metrics in the following categories (all prefixed with `vllm:`):
- Request metrics
- Performance metrics
- Resource usage
- Scheduler metrics
- Disaggregation metrics (when enabled)

**Note:** Specific metrics are subject to change between vLLM versions. Always refer to the [official documentation](https://docs.vllm.ai/en/latest/design/metrics.html) or inspect the `/metrics` endpoint for your vLLM version.

## Enabling Metrics in Dynamo

vLLM metrics are automatically exposed when running vLLM through Dynamo with metrics enabled.

## Inspecting Metrics

To see the actual metrics available in your vLLM version:

### 1. Launch vLLM with Metrics Enabled

```bash
# Set environment variables
export DYN_SYSTEM_ENABLED=true
export DYN_SYSTEM_PORT=8081

# Start vLLM worker (metrics enabled by default via --disable-log-stats=false)
python -m dynamo.vllm --model <model_name>

# Wait for engine to initialize
```

Metrics will be available at: `http://localhost:8081/metrics`

### 2. Fetch Metrics via curl

```bash
curl http://localhost:8081/metrics | grep "^vllm:"
```

### 3. Example Output

**Note:** The specific metrics shown below are examples and may vary depending on your vLLM version. Always inspect your actual `/metrics` endpoint for the current list.

```
# HELP vllm:request_success_total Number of successfully finished requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B"} 15.0
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B"} 150.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B"} 5.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B"} 165.0
vllm:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B"} 89.38
```

## Implementation Details

- vLLM v1 uses multiprocess metrics collection via `prometheus_client.multiprocess`
- `PROMETHEUS_MULTIPROC_DIR`: vLLM sets this environment variable to a temporary directory where multiprocess metrics are stored as memory-mapped files. Each worker process writes its metrics to separate files in this directory, which are aggregated when `/metrics` is scraped.
- Metrics are filtered by the `vllm:` prefix before being exposed
- The integration uses Dynamo's `register_engine_metrics_callback()` function
- Metrics appear after vLLM engine initialization completes
- vLLM v1 metrics are different from v0 - see the [official documentation](https://docs.vllm.ai/en/latest/design/metrics.html) for migration details

## See Also

### vLLM Metrics
- [Official vLLM Metrics Design Documentation](https://docs.vllm.ai/en/latest/design/metrics.html)
- [vLLM Production Metrics User Guide](https://docs.vllm.ai/en/latest/user/production_metrics.html)
- [vLLM GitHub - Metrics Implementation](https://github.com/vllm-project/vllm/tree/main/vllm/engine/metrics)

### Dynamo Metrics
- **Dynamo Metrics Guide**: See `docs/guides/metrics.md` for complete documentation on Dynamo runtime metrics
- **Dynamo Runtime Metrics**: Metrics prefixed with `dynamo_*` for runtime, components, endpoints, and namespaces
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Available at the same `/metrics` endpoint alongside vLLM metrics
- **Integration Code**: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
