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

# Dynamo Metrics Guide

This guide covers the metrics system in Dynamo, which provides hierarchical Prometheus metrics with automatic labeling and namespace organization. The metrics system enables observability across distributed inference workloads.

## Overview

Dynamo's metrics system is built around a hierarchical registry framework that automatically organizes metrics by namespace, component, and endpoint. This provides structured observability across the distributed runtime system.

## Architecture

### Hierarchical Metrics Registry

The metrics system follows a hierarchical structure:

```
DistributedRuntime (DRT)
├── Namespace
│   ├── Component
│   │   └── Endpoint
│   └── Component
│       └── Endpoint
└── Namespace
    └── Component
        └── Endpoint
```

Each level in the hierarchy can create and manage metrics, with automatic namespace prefixing and labeling based on the hierarchy.

### Key Components

- **MetricsRegistry**: Unified interface for creating and managing Prometheus metrics
- **Prometheus Integration**: Native Prometheus metric types support
- **Auto-labeling**: Automatic namespace, component, and endpoint labels
- **Metrics Endpoint**: The `/metrics` HTTP endpoint is exposed when `DYN_SYSTEM_ENABLED` is `true`, allowing Prometheus to scrape metrics. Configure the port with `DYN_SYSTEM_PORT`.




## Metric Types

Major Prometheus metric types that are available via MetricsRegistry:

### Basic Metrics
- **Counter**: Monotonically increasing values (e.g., request counts)
- **Gauge**: Values that can go up and down (e.g., active connections)
- **Histogram**: Distributions of values (e.g., request durations)
- **IntCounter**: Integer counters
- **IntGauge**: Integer gauges

### Vector Metrics (with dynamic labels)
- **CounterVec**: Counters with dynamic labels
- **IntCounterVec**: Integer counters with dynamic labels
- **IntGaugeVec**: Integer gauges with dynamic labels

## Base Metrics

Dynamo automatically provides a set of base metrics that are inherited for free when using the DistributedRuntime and request handlers. The followings are implemented today:

- **Request counters**: Total requests processed
- **Request duration**: Processing time histograms
- **Concurrent request tracking**: Current active requests
- **Error tracking**: Detailed error counts by type (deserialization, invalid messages, response streams, generation, publishing)

These base metrics are automatically created when using the Distributedruntime code that have request handlers. When an endpoint calls the ingress-or-handler function, these metrics are automatically measured and updated. Additional base metrics are being added to the system to expand the default observability coverage.

Base metrics are automatically exposed on the HTTP `/metrics` endpoint when the `DYN_SYSTEM_ENABLED` environment variable is set to `true`. This allows Prometheus and other monitoring tools to scrape the metrics without additional configuration.

The metrics endpoint port can be configured using the `DYN_SYSTEM_PORT` environment variable. If set to 0, the system will assign a random available port.

## Prometheus Output Example

Below is an example of how these metrics appear in Prometheus output on the HTTP /metrics endpoint:

```prometheus
# HELP dynamo_requests_total Total requests processed
# TYPE dynamo_requests_total counter
dynamo_requests_total{component="backend",endpoint="generate",namespace="dynamo"} 150

# HELP dynamo_request_duration_seconds Request processing time
# TYPE dynamo_request_duration_seconds histogram
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="0.1"} 120
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="0.5"} 145
dynamo_request_duration_seconds_bucket{component="backend",endpoint="generate",namespace="dynamo",le="1.0"} 150
dynamo_request_duration_seconds_sum{component="backend",endpoint="generate",namespace="dynamo"} 45.2
dynamo_request_duration_seconds_count{component="backend",endpoint="generate",namespace="dynamo"} 150

# HELP dynamo_concurrent_requests Current active requests
# TYPE dynamo_concurrent_requests gauge
dynamo_concurrent_requests{component="backend",endpoint="generate",namespace="dynamo"} 5

# HELP dynamo_errors_total Total errors by type
# TYPE dynamo_errors_total counter
dynamo_errors_total{component="backend",endpoint="generate",error_type="generate",namespace="dynamo"} 2
```

The metrics system automatically adds labels based on the hierarchy:
- **namespace**: The namespace name
- **component**: The component name (if applicable)
- **endpoint**: The endpoint name (if applicable)

### Error Types Tracked

The error counter automatically tracks these error types:
- `deserialization`: Failed to deserialize request data
- `invalid_message`: Unexpected message format
- `response_stream`: Failed to create response stream
- `generate`: Error during request generation
- `publish_response`: Failed to publish response data
- `publish_final`: Failed to publish final message

## Code Examples

### Creating Metrics at Different Hierarchy Levels

```rust
use dynamo_runtime::{DistributedRuntime, Runtime, Result};

// Create a distributed runtime
let rt = Runtime::from_current().unwrap();
let drt = DistributedRuntime::from_settings(rt).await.unwrap();

let namespace = drt.namespace("dynamo").unwrap();
let component = namespace.component("auth_service").unwrap();
// Create endpoint-level metrics
let endpoint = component.endpoint("login");
let login_attempts = endpoint.create_counter("login_attempts", "Login attempts", &[])?;
let login_success_rate = endpoint.create_gauge("login_success_rate", "Login success rate", &[])?;
```

### Using Vector Metrics with Dynamic Labels

```rust
// Create a CounterVec with dynamic labels
let http_requests = component.create_countervec(
    "http_requests_total",
    "HTTP requests by method and status",
    &["method", "status"],  // Dynamic label names
    &[("service", "api")]   // Static labels
)?;

// Use the vector with different label values
http_requests.with_label_values(&["GET", "200"]).inc_by(1.0);
http_requests.with_label_values(&["POST", "201"]).inc_by(1.0);
http_requests.with_label_values(&["GET", "404"]).inc_by(1.0);

// Create an IntGaugeVec for connection counts
let connections = endpoint.create_intgaugevec(
    "connections_active",
    "Active connections by instance",
    &["instance", "type"],
    &[("service", "api")]
)?;

connections.with_label_values(&["server1", "websocket"]).set(10);
connections.with_label_values(&["server2", "http"]).set(25);
```

### Working with Histograms

```rust
// Create histogram with custom buckets
let request_duration = namespace.create_histogram(
    "request_duration_seconds",
    "Request duration distribution",
    &[],
    Some(vec![0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
)?;

// Record observations
request_duration.observe(0.15);
request_duration.observe(0.8);
request_duration.observe(2.5);
```

## HTTP Server Metrics

The HTTP server automatically exposes metrics with the `dynamo_system_server` prefix:

```rust
// HTTP server uptime metric
dynamo_system_server_uptime_seconds{namespace="dynamo"} 3600
```

## Best Practices

### 1. Leverage Hierarchy

Create metrics at the appropriate hierarchy level:

```rust
// Service-specific metrics at namespace level
let api_requests = namespace.create_counter("requests_total", "API service requests", &[])?;

// Component-specific metrics at component level
let auth_requests = component.create_counter("requests_total", "Authentication requests", &[])?;

// Endpoint-specific metrics at endpoint level
let login_requests = endpoint.create_counter("requests_total", "Login endpoint requests", &[])?;
```

### 2. Use Appropriate Metric Types

```rust
// Use Counter for monotonically increasing values
let request_count = namespace.create_counter("requests_total", "Total requests", &[])?;

// Use Gauge for values that can go up and down
let active_connections = namespace.create_gauge("active_connections", "Active connections", &[])?;

// Use Histogram for distributions
let request_duration = namespace.create_histogram(
    "request_duration_seconds",
    "Request duration",
    &[],
    Some(vec![0.1, 0.5, 1.0, 2.0, 5.0])
)?;
```

### 3. Use Vector Metrics for Dynamic Labeling

```rust
// Use CounterVec when you need dynamic labels
let http_requests = namespace.create_countervec(
    "http_requests_total",
    "HTTP requests by method and status",
    &["method", "status"],
    &[]
)?;

// Use regular Counter for static metrics
let total_requests = namespace.create_counter("requests_total", "Total requests", &[])?;
```

## Monitoring and Visualization

### Prometheus Configuration

Configure Prometheus to scrape your metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'dynamo'
    static_configs:
      - targets: ['localhost:8080']
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
# Grafana: http://localhost:3001 (dynamo/dynamo)
# Prometheus: http://localhost:9090
```

## Troubleshooting

### Common Issues

1. **Metric Name Validation**
   ```rust
   // Invalid characters are automatically sanitized
   namespace.create_counter("request-count", "Requests", &[])?;  // becomes "requestcount"

   // Use valid names for clarity
   namespace.create_counter("request_count", "Requests", &[])?;
   ```

2. **Label Conflicts**
   ```rust
   // These will fail - "namespace", "component", and "endpoint" are reserved
   namespace.create_counter("requests", "Requests", &[("namespace", "custom")])?;
   namespace.create_counter("requests", "Requests", &[("component", "custom")])?;
   namespace.create_counter("requests", "Requests", &[("endpoint", "custom")])?;

   // Use different label names
   namespace.create_counter("requests", "Requests", &[("service", "custom")])?;
   ```

3. **Vector Metric Configuration**
   ```rust
   // This will fail - CounterVec requires label names
   component.create_countervec("requests", "Requests", &[], &[])?;

   // Provide label names
   component.create_countervec("requests", "Requests", &["method"], &[])?;
   ```

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
let custom_buckets = vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0];
let latency = namespace.create_histogram(
    "request_latency_seconds",
    "Request latency distribution",
    &[],
    Some(custom_buckets)
)?;
```

### Metric Aggregation

Metrics are automatically aggregated across hierarchy levels:

```rust
// Create metrics at different levels
let ns_counter = namespace.create_counter("requests", "Namespace requests", &[])?;
let comp_counter = component.create_counter("requests", "Component requests", &[])?;
let endpoint_counter = endpoint.create_counter("requests", "Endpoint requests", &[])?;

// All will be visible in the final Prometheus output
let all_metrics = drt.prometheus_metrics_fmt()?;
```

This metrics system provides observability capabilities for monitoring and debugging Dynamo applications at scale.

## Related Documentation

- [Backend Guide](backend.md) - Creating Python workers with metrics
- [KV Router Performance Tuning](kv_router_perf_tuning.md) - Performance optimization with metrics
- [Disaggregation Performance Tuning](disagg_perf_tuning.md) - Distributed inference metrics
- [Deploy Metrics README](../../deploy/metrics/README.md) - Monitoring stack setup and deployment