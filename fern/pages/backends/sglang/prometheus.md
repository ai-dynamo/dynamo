---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# SGLang Prometheus Metrics

## Overview

When running SGLang through Dynamo, SGLang engine metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both SGLang engine metrics (prefixed with `sglang:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

**For the complete and authoritative list of all SGLang metrics**, always refer to the [official SGLang Production Metrics documentation](https://docs.sglang.io/references/production_metrics.html).

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

Launch a frontend and SGLang backend to test metrics:

```bash
# Start frontend (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
$ python -m dynamo.frontend

# Enable system metrics server on port 8081
$ DYN_SYSTEM_PORT=8081 python -m dynamo.sglang --model <model_name> --enable-metrics
```

Wait for the SGLang worker to start, then send requests and check metrics:

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
curl -s localhost:8081/metrics | grep "^sglang:"
```

## Exposed Metrics

SGLang exposes metrics in Prometheus Exposition Format text at the `/metrics` HTTP endpoint. All SGLang engine metrics use the `sglang:` prefix and include labels (e.g., `model_name`, `engine_type`, `tp_rank`, `pp_rank`) to identify the source.

**Example Prometheus Exposition Format text:**

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

**Note:** The specific metrics shown above are examples and may vary depending on your SGLang version. Always inspect your actual `/metrics` endpoint or refer to the [official documentation](https://docs.sglang.io/references/production_metrics.html) for the current list.

### Metric Categories

SGLang provides metrics in the following categories (all prefixed with `sglang:`):

- **Throughput metrics** - Token processing rates
- **Resource usage** - System resource consumption
- **Latency metrics** - Request and token latency measurements
- **Disaggregation metrics** - Metrics specific to disaggregated deployments (when enabled)

**Note:** Specific metrics are subject to change between SGLang versions. Always refer to the [official documentation](https://docs.sglang.io/references/production_metrics.html) or inspect the `/metrics` endpoint for your SGLang version.

## Available Metrics

The official SGLang documentation includes complete metric definitions with:
- HELP and TYPE descriptions
- Counter, Gauge, and Histogram metric types
- Metric labels (e.g., `model_name`, `engine_type`, `tp_rank`, `pp_rank`)
- Setup guide for Prometheus + Grafana monitoring
- Troubleshooting tips and configuration examples

For the complete and authoritative list of all SGLang metrics, see the [official SGLang Production Metrics documentation](https://docs.sglang.io/references/production_metrics.html).

## Implementation Details

- SGLang uses multiprocess metrics collection via `prometheus_client.multiprocess.MultiProcessCollector`
- Metrics are filtered by the `sglang:` prefix before being exposed
- The integration uses Dynamo's `register_engine_metrics_callback()` function
- Metrics appear after SGLang engine initialization completes

## Related Documentation

### SGLang Metrics
- [Official SGLang Production Metrics](https://docs.sglang.io/references/production_metrics.html)
- [SGLang GitHub - Metrics Collector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/metrics/collector.py)

### Dynamo Metrics
- [Dynamo Metrics Guide](../../observability/metrics.md) - Complete documentation on Dynamo runtime metrics
- [Prometheus and Grafana Setup](../../observability/prometheus-grafana.md) - Visualization setup instructions
- Dynamo runtime metrics (prefixed with `dynamo_*`) are available at the same `/metrics` endpoint alongside SGLang metrics
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Integration code: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
