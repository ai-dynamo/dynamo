# SGLang Prometheus Metrics

**📚 Official Documentation**: [SGLang Production Metrics](https://docs.sglang.ai/references/production_metrics.html)

This document describes how SGLang Prometheus metrics are exposed in Dynamo.

## Overview

When running SGLang through Dynamo, SGLang engine metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both SGLang engine metrics (prefixed with `sglang:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

For the complete and authoritative list of all SGLang metrics, always refer to the official documentation linked above.

Dynamo runtime metrics are documented in [docs/observability/metrics.md](../../observability/metrics.md).

## Metric Reference

The official documentation includes:
- Complete metric definitions with HELP and TYPE descriptions
- Example metric output in Prometheus exposition format
- Counter, Gauge, and Histogram metrics
- Metric labels (e.g., `model_name`, `engine_type`, `tp_rank`, `pp_rank`)
- Setup guide for Prometheus + Grafana monitoring
- Troubleshooting tips and configuration examples

## Metric Categories

SGLang provides metrics in the following categories (all prefixed with `sglang:`):
- Throughput metrics
- Resource usage
- Latency metrics
- Disaggregation metrics (when enabled)

**Note:** Specific metrics are subject to change between SGLang versions. Always refer to the [official documentation](https://docs.sglang.ai/references/production_metrics.html) or inspect the `/metrics` endpoint for your SGLang version.

## Enabling Metrics in Dynamo

SGLang metrics are automatically exposed when running SGLang through Dynamo with metrics enabled.

## Inspecting Metrics

To see the actual metrics available in your SGLang version:

### 1. Launch SGLang with Metrics Enabled

```bash
# Set environment variables
export DYN_SYSTEM_ENABLED=true
export DYN_SYSTEM_PORT=8081

# Start SGLang worker with metrics enabled
python -m dynamo.sglang --model <model_name> --enable-metrics

# Wait for engine to initialize
```

Metrics will be available at: `http://localhost:8081/metrics`

### 2. Fetch Metrics via curl

```bash
curl http://localhost:8081/metrics | grep "^sglang:"
```

### 3. Example Output

**Note:** The specific metrics shown below are examples and may vary depending on your SGLang version. Always inspect your actual `/metrics` endpoint for the current list.

```
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8128902.0
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7557572.0
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0075
```

## Implementation Details

- SGLang uses multiprocess metrics collection via `prometheus_client.multiprocess.MultiProcessCollector`
- Metrics are filtered by the `sglang:` prefix before being exposed
- The integration uses Dynamo's `register_engine_metrics_callback()` function
- Metrics appear after SGLang engine initialization completes

## See Also

### SGLang Metrics
- [Official SGLang Production Metrics](https://docs.sglang.ai/references/production_metrics.html)
- [SGLang GitHub - Metrics Collector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/metrics/collector.py)

### Dynamo Metrics
- **Dynamo Metrics Guide**: See [docs/observability/metrics.md](../../observability/metrics.md) for complete documentation on Dynamo runtime metrics
- **Dynamo Runtime Metrics**: Metrics prefixed with `dynamo_*` for runtime, components, endpoints, and namespaces
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Available at the same `/metrics` endpoint alongside SGLang metrics
- **Integration Code**: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
