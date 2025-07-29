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

## Creating Metrics at Different Hierarchy Levels

### Runtime-Level Metrics

```rust
use dynamo_runtime::DistributedRuntime;

let runtime = DistributedRuntime::new()?;
let namespace = runtime.namespace("my_namespace")?;
let component = namespace.component("my_component")?;
let endpoint = component.endpoint("my_endpoint")?;

// Create endpoint-level counters (this is a Prometheus Counter type)
let total_requests = endpoint.create_counter(
    "total_requests",
    "Total requests across all namespaces",
    &[]
)?;

let active_connections = endpoint.create_gauge(
    "active_connections",
    "Number of active client connections",
    &[]
)?;
```

### Namespace-Level Metrics

```rust
let namespace = runtime.namespace("my_model")?;

// Namespace-scoped metrics
let model_requests = namespace.create_counter(
    "model_requests",
    "Requests for this specific model",
    &[]
)?;

let model_latency = namespace.create_histogram(
    "model_latency_seconds",
    "Model inference latency",
    &[],
    &[0.001, 0.01, 0.1, 1.0, 10.0]
)?;
```

### Component-Level Metrics

```rust
let component = namespace.component("backend")?;

// Component-specific metrics
let backend_requests = component.create_counter(
    "backend_requests",
    "Requests handled by this backend component",
    &[]
)?;

let gpu_memory_usage = component.create_gauge(
    "gpu_memory_bytes",
    "GPU memory usage in bytes",
    &[]
)?;
```

### Endpoint-Level Metrics

```rust
let endpoint = component.endpoint("generate")?;

// Endpoint-specific metrics
let generate_requests = endpoint.create_counter(
    "generate_requests",
    "Generate endpoint requests",
    &[]
)?;

let generate_latency = endpoint.create_histogram(
    "generate_latency_seconds",
    "Generate endpoint latency",
    &[],
    &[0.001, 0.01, 0.1, 1.0, 10.0]
)?;
```

## Creating Vector Metrics with Dynamic Labels

Use vector metrics when you need to track metrics with different label values:

```rust
// Counter with labels
let requests_by_model = endpoint.create_counter_vec(
    "requests_by_model",
    "Requests by model type",
    &["model_type", "model_size"]
)?;

// Increment with specific labels
requests_by_model.with_label_values(&["llama", "7b"]).inc();
requests_by_model.with_label_values(&["gpt", "13b"]).inc();

// Gauge with labels
let memory_by_gpu = component.create_gauge_vec(
    "gpu_memory_bytes",
    "GPU memory usage by device",
    &["gpu_id", "memory_type"]
)?;

memory_by_gpu.with_label_values(&["0", "allocated"]).set(8192.0);
memory_by_gpu.with_label_values(&["0", "cached"]).set(4096.0);
```

## Creating Histograms

Histograms are useful for measuring distributions of values like latency:

```rust
let latency_histogram = endpoint.create_histogram(
    "request_latency_seconds",
    "Request latency distribution",
    &[],
    &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)?;

// Record latency values
latency_histogram.observe(0.023); // 23ms
latency_histogram.observe(0.156); // 156ms
```

## Transitioning from Plain Prometheus

If you're currently using plain Prometheus metrics, transitioning to Dynamo's `MetricsRegistry` is straightforward:

### Before (Plain Prometheus)

```rust
use prometheus::{Counter, Opts, Registry};

// Create a registry to hold metrics
let registry = Registry::new();
let counter_opts = Opts::new("my_counter", "My custom counter");
let counter = Counter::with_opts(counter_opts).unwrap();
registry.register(Box::new(counter.clone())).unwrap();

// Use the counter
counter.inc();

// To expose metrics, you'd need to set up an HTTP server manually
// and implement the /metrics endpoint yourself
```

### After (Dynamo MetricsRegistry)

```rust
let counter = endpoint.create_counter(
    "my_counter",
    "My custom counter",
    &[]
)?;

counter.inc();
```

**Note:** The metric is automatically registered when created via the endpoint's `create_counter` factory method.

**Benefits of Dynamo's approach:**
- **Automatic registration**: Metrics created via endpoint's `create_*` factory methods are automatically registered with the system
- Automatic labeling with namespace, component, and endpoint information
- Consistent metric naming with `dynamo_` prefix
- Built-in HTTP metrics endpoint when enabled with `DYN_SYSTEM_ENABLED=true`
- Hierarchical metric organization

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

## Advanced Features

### Custom Buckets for Histograms

```rust
// Define custom buckets for your use case
let custom_buckets = vec![0.001, 0.01, 0.1, 1.0, 10.0];
let latency = endpoint.create_histogram(
    "api_latency_seconds",
    "API latency in seconds",
    &[],
    &custom_buckets
)?;
```

### Metric Aggregation

```rust
// Aggregate metrics across multiple endpoints
let total_requests = namespace.create_counter(
    "total_requests",
    "Total requests across all endpoints",
    &[]
)?;
```

## Related Documentation

- [Distributed Runtime Architecture](../architecture/distributed_runtime.md)
- [Dynamo Architecture Overview](../architecture/architecture.md)
- [Dynamo Flow](../architecture/dynamo_flow.md)
- [Backend Guide](backend.md)
- [Dynamo Run Guide](dynamo_run.md)
- [Performance Tuning Guides](kv_router_perf_tuning.md)