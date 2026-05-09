---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Prometheus
---

## Overview

When running vLLM through Dynamo, vLLM engine metrics are automatically passed through and exposed on Dynamo's `/metrics` endpoint (default port 8081). This allows you to access both vLLM engine metrics (prefixed with `vllm:`) and Dynamo runtime metrics (prefixed with `dynamo_*`) from a single worker backend endpoint.

**For the complete and authoritative list of all vLLM metrics**, always refer to the [official vLLM Metrics Design documentation](https://docs.vllm.ai/en/stable/design/metrics.html).

**For LMCache metrics and integration**, see the [LMCache Integration Guide](../../integrations/lmcache-integration.md).

**For Dynamo runtime metrics**, see the [Dynamo Metrics Guide](../../observability/metrics.md).

**For visualization setup instructions**, see the [Prometheus and Grafana Setup Guide](../../observability/prometheus-grafana.md).

## Environment Variables and Flags

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_SYSTEM_PORT` | System metrics/health port. Required to expose `/metrics` endpoint. | `-1` (disabled) | `8081` |

## Getting Started Quickly

This is a single machine example.

### Start Observability Stack

For visualizing metrics with Prometheus and Grafana, start the observability stack. See [Observability Getting Started](../../observability/README.md#getting-started-quickly) for instructions.

### Launch Dynamo Components

The launch scripts in `examples/backends/vllm/launch/` already enable metrics on port 8081 by default. For example:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

Once the deployment is running, send a request and check metrics:

```bash
curl -s localhost:8081/metrics | grep "^vllm:"
```

## Exposed Metrics

vLLM exposes metrics in Prometheus Exposition Format text at the `/metrics` HTTP endpoint. All vLLM engine metrics use the `vllm:` prefix and include labels (e.g., `model_name`, `finished_reason`, `scheduling_event`) to identify the source.

**Example Prometheus Exposition Format text:**

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

**Note:** The specific metrics shown above are examples and may vary depending on your vLLM version. Always inspect your actual `/metrics` endpoint or refer to the [official documentation](https://docs.vllm.ai/en/stable/design/metrics.html) for the current list.

### Metric Categories

vLLM provides metrics in the following categories (all prefixed with `vllm:`):

- **Request metrics** - Request success, failure, and completion tracking
- **Performance metrics** - Latency, throughput, and timing measurements
- **Resource usage** - System resource consumption
- **Scheduler metrics** - Scheduling and queue management
- **Disaggregation metrics** - Metrics specific to disaggregated deployments (when enabled)

**Note:** Specific metrics are subject to change between vLLM versions. Always refer to the [official documentation](https://docs.vllm.ai/en/stable/design/metrics.html) or inspect the `/metrics` endpoint for your vLLM version.

## Available Metrics

The official vLLM documentation includes complete metric definitions with:
- Detailed explanations and design rationale
- Counter, Gauge, and Histogram metric types
- Metric labels (e.g., `model_name`, `finished_reason`, `scheduling_event`)
- Information about v1 metrics migration
- Future work and deprecated metrics

For the complete and authoritative list of all vLLM metrics, see the [official vLLM Metrics Design documentation](https://docs.vllm.ai/en/stable/design/metrics.html).

## LMCache Metrics

When LMCache is enabled, LMCache metrics (prefixed with `lmcache:`) are automatically exposed via Dynamo's `/metrics` endpoint alongside vLLM and Dynamo metrics.

To try it out, use the LMCache launch script:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg_lmcache.sh
```

Send a request and view LMCache metrics:

```bash
curl -s localhost:8081/metrics | grep "^lmcache:"
```

### Troubleshooting

Troubleshooting LMCache-related metrics and logs (including `PrometheusLogger instance already created with different metadata` and `PROMETHEUS_MULTIPROC_DIR` warnings) is documented in:

- [LMCache Integration Guide](../../integrations/lmcache-integration.md#troubleshooting)

**For complete LMCache configuration and metric details**, see:
- [LMCache Integration Guide](../../integrations/lmcache-integration.md) - Setup and configuration
- [LMCache Observability Documentation](https://docs.lmcache.ai/production/observability/vllm_endpoint.html) - Complete metrics reference

## Implementation Details

- vLLM v1 uses multiprocess metrics collection via `prometheus_client.multiprocess`
- `PROMETHEUS_MULTIPROC_DIR`: (optional). By default, Dynamo automatically manages this environment variable, setting it to a temporary directory where multiprocess metrics are stored as memory-mapped files. Each worker process writes its metrics to separate files in this directory, which are aggregated when `/metrics` is scraped. Users only need to set this explicitly where complete control over the metrics directory is required.
- Dynamo uses `MultiProcessCollector` to aggregate metrics from all worker processes
- Metrics are filtered by the `vllm:` and `lmcache:` prefixes before being exposed (when LMCache is enabled)
- The integration uses Dynamo's `register_engine_metrics_callback()` function with the global `REGISTRY`
- Metrics appear after vLLM engine initialization completes
- vLLM v1 metrics are different from v0 - see the [official documentation](https://docs.vllm.ai/en/stable/design/metrics.html) for migration details

## Related Documentation

### vLLM Metrics
- [Official vLLM Metrics Design Documentation](https://docs.vllm.ai/en/stable/design/metrics.html)
- [vLLM Production Metrics User Guide](https://docs.vllm.ai/en/stable/usage/metrics.html)
- [vLLM GitHub - Metrics Implementation](https://github.com/vllm-project/vllm/tree/main/vllm/v1/metrics)

### Dynamo Metrics
- [Dynamo Metrics Guide](../../observability/metrics.md) - Complete documentation on Dynamo runtime metrics
- [Prometheus and Grafana Setup](../../observability/prometheus-grafana.md) - Visualization setup instructions
- Dynamo runtime metrics (prefixed with `dynamo_*`) are available at the same `/metrics` endpoint alongside vLLM metrics
  - Implementation: `lib/runtime/src/metrics.rs` (Rust runtime metrics)
  - Metric names: `lib/runtime/src/metrics/prometheus_names.rs` (metric name constants)
  - Integration code: `components/src/dynamo/common/utils/prometheus.py` - Prometheus utilities and callback registration
