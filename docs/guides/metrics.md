<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo `MetricsRegistry`

## Overview

Dynamo provides built-in metrics capabilities through the `MetricsRegistry` trait, which is automatically available whenever you use the `DistributedRuntime` framework. This guide explains how to use metrics for observability and monitoring across all Dynamo components and services.

The `MetricsRegistry` trait is implemented by `DistributedRuntime`, `Namespace`, `Component`, and `Endpoint`, providing a hierarchical approach to metric collection that matches Dynamo's distributed architecture:

- `DistributedRuntime`: Global metrics across the entire runtime
- `Namespace`: Metrics scoped to a specific namespace
- `Component`: Metrics for a specific component within a namespace
- `Endpoint`: Metrics for individual endpoints within a component

This hierarchical structure allows you to create metrics at the appropriate level of granularity for your monitoring needs.

**Worker Metrics**: Backend metrics automatically include labels for `namespace`, `component`, and `endpoint` based on the hierarchy level where they're created. Additional custom labels can be added at the implementor's discretion and will appear in the Prometheus output alongside the automatic labels.

**Worker Metrics**: The core Dynamo backend system automatically exposes the following worker metrics with automatic `namespace`, `component`, and `endpoint` labeling:

- `dynamo_concurrent_requests`: Number of requests currently being processed by work handler (gauge)
- `dynamo_request_bytes_total`: Total number of bytes received in requests by work handler (counter)
- `dynamo_request_duration_seconds`: Time spent processing requests by work handler (histogram)
- `dynamo_requests_total`: Total number of requests processed by work handler (counter)
- `dynamo_response_bytes_total`: Total number of bytes sent in responses by work handler (counter)
- `dynamo_system_uptime_seconds`: Total uptime of the DistributedRuntime in seconds (gauge)

These metrics include labels for `namespace`, `component`, and `endpoint` to provide detailed observability into backend worker performance, request processing, and system health. Example label values include `namespace="dynamo"`, `component="backend"`, and `endpoint="generate"`.

**Frontend Metrics**: Frontend labels and metrics are implemented separately from the core Dynamo metrics system. When using Dynamo HTTP Frontend (available with `--framework VLLM` or `--framework TENSORRTLLM`), the following metrics are automatically exposed:

- `nv_llm_http_service_inflight_requests`: Number of inflight requests (gauge)
- `nv_llm_http_service_input_sequence_tokens`: Input sequence length in tokens (histogram)
- `nv_llm_http_service_inter_token_latency_seconds`: Inter-token latency in seconds (histogram)
- `nv_llm_http_service_output_sequence_tokens`: Output sequence length in tokens (histogram)
- `nv_llm_http_service_request_duration_seconds`: Duration of LLM requests (histogram)
- `nv_llm_http_service_requests_total`: Total number of LLM requests processed (counter)
- `nv_llm_http_service_time_to_first_token_seconds`: Time to first token in seconds (histogram)

These metrics include labels for `model` to provide detailed observability into the HTTP frontend performance. Some metrics also include additional labels such as `endpoint`, `request_type`, and `status`. Histogram metrics also include `le` (less than or equal) labels for bucket boundaries. Example label values include `model="qwen/qwen3-0.6b"`, `endpoint="chat_completions"`, `request_type="stream"`, and `status="success"`.

### Automatic Metrics

When you enable the metrics HTTP endpoint with `DYN_SYSTEM_ENABLED=true`, Dynamo automatically adds:

- `dynamo_system_uptime_seconds`: System uptime counter
- HTTP server metrics for the metrics endpoint itself

## Environment Configuration

Enable the metrics HTTP endpoint:

```bash
export DYN_SYSTEM_ENABLED=true
export DYN_SYSTEM_PORT=8081  # Use 0 for random port assignment
```

The `DYN_SYSTEM_PORT=0` assigns a random port, which is useful for integration testing to avoid port conflicts.

## Implementation Examples

For detailed implementation examples showing how to create metrics at different hierarchy levels, create vector metrics with dynamic labels, and transition from plain Prometheus, see the [Implementation Examples section](../../deploy/metrics/README.md#implementation-examples) in the deploy metrics README.

## Prometheus Output Example

To enable metrics, launch your Dynamo service with the required environment variables:

```bash
# Launch dynamo.vllm with metrics enabled (example):
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model-path /path/to/model
```

Then when you make the curl call to the metrics endpoint:

```bash
curl http://localhost:8081/metrics
```

You'll see output like this:

```
# HELP dynamo_my_counter My custom counter
# TYPE dynamo_my_counter counter
dynamo_my_counter{namespace="dynamo",component="backend",endpoint="generate"} 42

# HELP dynamo_system_uptime_seconds System uptime
# TYPE dynamo_system_uptime_seconds counter
dynamo_system_uptime_seconds{namespace="dynamo"} 42
```

## Monitoring and Visualization

### Prometheus Configuration

Configure Prometheus to scrape your metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'dynamo'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboards

Use the provided Grafana dashboards in `deploy/metrics/grafana_dashboards/`:

- `grafana-dynamo-dashboard.json`: General Dynamo dashboard for SW and HW metrics
- `grafana-dcgm-metrics.json`: DCGM GPU metrics dashboard

### Starting the Monitoring Stack

```bash
# Start Prometheus and Grafana
docker compose -f deploy/docker-compose.yml --profile metrics up -d

# Access the dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Troubleshooting

Common issues and solutions:

1. **Metrics not appearing**: Check that `DYN_SYSTEM_ENABLED=true`
2. **Port conflicts**: Use `DYN_SYSTEM_PORT=0` for random port assignment
3. **Missing labels**: Ensure you're using the correct hierarchy level

### Debugging

Enable debug logging to see metric registration details:

```rust
// Check metric output
let metrics = namespace.prometheus_metrics_fmt()?;
println!("Metrics: {}", metrics);
```

### Verification Steps

1. **Verify services are running:**
   ```bash
   docker compose ps
   ```

2. **Check logs:**
   ```bash
   docker compose logs prometheus
   docker compose logs grafana
   ```

3. **Test metrics endpoint:**
   ```bash
   curl http://localhost:8081/metrics
   ```

## Related Documentation

- [Distributed Runtime Architecture](../architecture/distributed_runtime.md)
- [Dynamo Architecture Overview](../architecture/architecture.md)
- [Dynamo Flow](../architecture/dynamo_flow.md)
- [Backend Guide](backend.md)
- [Dynamo Run Guide](dynamo_run.md)
- [Performance Tuning Guides](kv_router_perf_tuning.md)
- [Metrics Implementation Examples](../../deploy/metrics/README.md#implementation-examples)