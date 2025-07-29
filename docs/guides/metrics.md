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

# Dynamo MetricsRegistry Guide

This guide covers the MetricsRegistry in Dynamo, which provides hierarchical Prometheus metrics with automatic labeling and namespace organization. The MetricsRegistry enables observability across distributed inference workloads.

**MetricsRegistry is a common universal Dynamo built-in** that provides standardized observability capabilities across all Dynamo components and services. It's automatically available whenever you use the DistributedRuntime framework.

## Overview

Dynamo's MetricsRegistry is built around a hierarchical registry framework that automatically organizes metrics by namespace, component, and endpoint. This provides structured observability across the distributed runtime system.

**MetricsRegistry is a trait** that is implemented by the core distributed runtime components:
- **DistributedRuntime**: Root level metrics registry
- **Namespace**: Namespace-level metrics registry
- **Component**: Component-level metrics registry
- **Endpoint**: Endpoint-level metrics registry

Each level in the hierarchy implements the MetricsRegistry trait, allowing you to create and manage metrics at the appropriate level while maintaining automatic namespace prefixing and labeling.

## Transitioning from Raw Prometheus

One of the key benefits of Dynamo's MetricsRegistry is how easy it is to transition from raw Prometheus metrics to the distributed runtime's Prometheus constructors. The transition is seamless and requires minimal code changes.

### Before: Raw Prometheus

```rust
use prometheus::{Counter, Opts};

let opts = Opts::new("my_counter", "A simple counter");
let counter = Counter::with_opts(opts).unwrap();  // Prometheus counter
```

### After: Dynamo MetricsRegistry

```rust
use dynamo_runtime::MetricsRegistry;

let counter = endpoint.create_intcounter("my_counter", "A simple counter", &[])?;  // Prometheus counter
```

**All the rest of your code can remain the same!** The counter still has the same API for incrementing, but now it's automatically:

- Exposed on the HTTP metrics endpoint
- Prefixed with the namespace (`dynamo_`)
- Labeled with namespace, component, and endpoint information
- Integrated into the distributed runtime's metrics collection

### Enabling the HTTP Metrics Endpoint

To expose your metrics via HTTP, simply enable the system endpoint with environment variables:

```bash
$ DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm ... ... &
```

Then access your metrics:

```bash
$ curl http://localhost:8081/metrics
# HELP dynamo_my_counter A simple counter
# TYPE dynamo_my_counter counter
dynamo_my_counter{component="example_component",endpoint="example_endpoint",namespace="dynamo"} 123
```

The metrics endpoint port can be configured using the `DYN_SYSTEM_PORT` environment variable. If set to 0, the system will assign a random available port, which is useful for integration testing to avoid port conflicts.

## Architecture

### Hierarchical Metrics Registry

The MetricsRegistry follows a hierarchical structure:

```
DistributedRuntime (DRT)
├── Namespace1
│   ├── Component1
│   │   └── Endpoint1
│   └── Component2
│       └── Endpoint2
└── Namespace2
    └── Component3
        └── Endpoint3
        ...
        └── EndpointN
```

### Automatic Labeling

The MetricsRegistry automatically adds labels based on the hierarchy, such as namespace, component, and endpoint. In addition, the `dynamo_system_uptime_seconds` metric is also automatically added to track system uptime.

## Code Examples

### Creating Metrics at Different Hierarchy Levels

```rust
use dynamo_runtime::{DistributedRuntime, Runtime, Result};

// Create a distributed runtime
let rt = Runtime::from_current().unwrap();
let drt = DistributedRuntime::from_settings(rt).await.unwrap();

let namespace = drt.namespace("dynamo").unwrap();
let component = namespace.component("auth_service").unwrap();
let endpoint = component.endpoint("login");
// Create two endpoint-level metrics:
let login_attempts = endpoint.create_counter("login_attempts", "Login attempts", &[])?;
let login_success_rate = endpoint.create_gauge("login_success_rate", "Login success rate", &[])?;
```

### Creating Vector Metrics with Dynamic Labels

```rust
// Create a CounterVec with dynamic labels
let http_requests = component.create_countervec(
    "http_requests_total",
    "Total HTTP requests",
    &["method", "status_code"],
    &[]
)?;

// Use the vector with specific label values
http_requests.with_label_values(&["GET", "200"]).inc();
http_requests.with_label_values(&["POST", "404"]).inc();
```

### Creating Histograms

```rust
// Create a histogram for request duration
let request_duration = endpoint.create_histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    &[],
    &[0.1, 0.5, 1.0, 2.0, 5.0]
)?;

// Record observations
request_duration.observe(0.25);
```

## Base Metrics

Dynamo automatically provides base metrics for all endpoints:

- `dynamo_requests_total`: Total number of requests
- `dynamo_request_duration_seconds`: Request duration histogram
- `dynamo_errors_total`: Total number of errors by type
- `dynamo_system_uptime_seconds`: System uptime

These base metrics are automatically created when using the Distributedruntime code that have request handlers. When an endpoint calls the request handler function, these metrics are automatically measured and updated. Additional base metrics are being added to the system to expand the default observability coverage.

## Prometheus Output Example

```prometheus
# HELP dynamo_requests_total Total requests
# TYPE dynamo_requests_total counter
dynamo_requests_total{component="backend",endpoint="generate",namespace="dynamo"} 1000

# HELP dynamo_request_duration_seconds Request duration
# TYPE dynamo_request_duration_seconds histogram
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="0.1"} 800
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="0.5"} 950
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="1.0"} 1000

# HELP dynamo_errors_total Total errors by type
# TYPE dynamo_errors_total counter
dynamo_errors_total{component="backend",endpoint="generate",error_type="generate",namespace="dynamo"} 2

# HTTP server uptime metric
dynamo_system_server_uptime_seconds{namespace="dynamo"} 3600
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
   curl http://localhost:8080/metrics
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

- [Distributed Runtime Guide](../distributed.md)
- [HTTP Service Guide](../http.md)
- [Component Architecture](../components.md)