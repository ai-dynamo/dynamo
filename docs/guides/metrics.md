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

# Dynamo MetricsRegistry

## Overview

Dynamo provides built-in metrics capabilities through the `MetricsRegistry` trait, which is automatically available whenever you use the `DistributedRuntime` framework. This guide explains how to use metrics for observability and monitoring across all Dynamo components.

## Automatic Metrics

Dynamo automatically exposes metrics with the `dynamo_` name prefixes. It also adds labels such as `dynamo_namespace`, `dynamo_component`, and `dynamo_endpoint` -- the prefix is needed as to not conflict with existing Kubernetes labels.

**Component Metrics**: The core Dynamo backend system automatically exposes metrics with the `dynamo_component_*` prefix. The followings are examples that currently exist for all components that use the `DistributedRuntime` framework. More metrics are being added and will appear in future releases:

- `dynamo_component_concurrent_requests`: Requests currently being processed (gauge)
- `dynamo_component_request_bytes_total`: Total bytes received in requests (counter)
- `dynamo_component_request_duration_seconds`: Request processing time (histogram)
- `dynamo_component_requests_total`: Total requests processed (counter)
- `dynamo_component_response_bytes_total`: Total bytes sent in responses (counter)
- `dynamo_component_system_uptime_seconds`: DistributedRuntime uptime (gauge)

**Specialized Component Metrics**: Some components expose additional metrics specific to their functionality. For example, the Preprocessor will have labels in the following convention:

- `dynamo_preprocessor_*`: Metrics specific to preprocessor components

**Frontend Metrics**: When using Dynamo HTTP Frontend (`--framework VLLM` or `--framework TENSORRTLLM`), these metrics are automatically exposed with the `dynamo_frontend_*` prefix. These metrics include `model` labels containing the model name:

- `dynamo_frontend_inflight_requests`: Inflight requests (gauge)
- `dynamo_frontend_input_sequence_tokens`: Input sequence length (histogram)
- `dynamo_frontend_inter_token_latency_seconds`: Inter-token latency (histogram)
- `dynamo_frontend_output_sequence_tokens`: Output sequence length (histogram)
- `dynamo_frontend_request_duration_seconds`: LLM request duration (histogram)
- `dynamo_frontend_requests_total`: Total LLM requests (counter)
- `dynamo_frontend_time_to_first_token_seconds`: Time to first token (histogram)

## Metrics Hierarchy

The `MetricsRegistry` trait is implemented by `DistributedRuntime`, `Namespace`, `Component`, and `Endpoint`, providing a hierarchical approach to metric collection that matches Dynamo's distributed architecture:

- `DistributedRuntime`: Global metrics across the entire runtime
- `Namespace`: Metrics scoped to a specific dynamo_namespace
- `Component`: Metrics for a specific dynamo_component within a namespace
- `Endpoint`: Metrics for individual dynamo_endpoint within a component

This hierarchical structure allows you to create metrics at the appropriate level of granularity for your monitoring needs.

## Environment Configuration

Enable metrics with environment variables:

```bash
export DYN_SYSTEM_ENABLED=true
export DYN_SYSTEM_PORT=8081  # Use 0 for random port
```

With `DYN_SYSTEM_ENABLED=true`, Dynamo automatically adds:
- `dynamo_system_uptime_seconds`: System uptime counter
- HTTP server metrics for the metrics endpoint

## Implementation Examples

See [Implementation Examples](../../deploy/metrics/README.md#implementation-examples) for detailed examples of creating metrics at different hierarchy levels and using dynamic labels.

## Prometheus Output Example

Launch with metrics enabled:
```bash
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model-path /path/to/model
```

Query metrics:
```bash
curl http://localhost:8081/metrics
```

Output:
```
# HELP dynamo_my_counter My custom counter
# TYPE dynamo_my_counter counter
dynamo_my_counter{dynamo_namespace="dynamo",dynamo_component="backend",dynamo_endpoint="generate"} 42

# HELP dynamo_system_uptime_seconds System uptime
# TYPE dynamo_system_uptime_seconds counter
dynamo_system_uptime_seconds{dynamo_namespace="dynamo"} 42
```

## Monitoring and Visualization

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'dynamo'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboards

Use dashboards in `deploy/metrics/grafana_dashboards/`:
- `grafana-dynamo-dashboard.json`: General Dynamo dashboard
- `grafana-dcgm-metrics.json`: DCGM GPU metrics dashboard

### Start Monitoring Stack

```bash
docker compose -f deploy/docker-compose.yml --profile metrics up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Troubleshooting

Common issues:
1. **Metrics not appearing**: Check `DYN_SYSTEM_ENABLED=true`
2. **Port conflicts**: Use `DYN_SYSTEM_PORT=0` for random port
3. **Missing labels**: Verify hierarchy level

### Debugging

```rust
let metrics = namespace.prometheus_metrics_fmt()?;
println!("Metrics: {}", metrics);
```

### Verification

```bash
docker compose ps
docker compose logs prometheus
curl http://localhost:8081/metrics
```

## Related Documentation

- [Distributed Runtime Architecture](../architecture/distributed_runtime.md)
- [Dynamo Architecture Overview](../architecture/architecture.md)
- [Backend Guide](backend.md)
- [Metrics Implementation Examples](../../deploy/metrics/README.md#implementation-examples)