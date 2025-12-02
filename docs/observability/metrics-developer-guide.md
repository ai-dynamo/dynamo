<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Metrics Developer Guide

This guide explains how to create and use custom metrics in Dynamo components using the Dynamo metrics API.

## Metrics Exposure

All metrics created via the Dynamo metrics API are automatically exposed on the `/metrics` HTTP endpoint in Prometheus Exposition Format text when the following environment variable is set:

- `DYN_SYSTEM_PORT=<port>` - Port for the metrics endpoint (set to positive value to enable, default: `-1` disabled)

Example:
```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model <model>
```

Prometheus Exposition Format text metrics will be available at: `http://localhost:8081/metrics`

## Metric Name Constants

The [prometheus_names.rs](../../lib/runtime/src/metrics/prometheus_names.rs) module provides centralized metric name constants and sanitization functions to ensure consistency across all Dynamo components.

---

## Metrics API in Rust

The metrics API is accessible through the `.metrics()` method on runtime, namespace, component, and endpoint objects. See [Runtime Hierarchy](metrics.md#runtime-hierarchy) for details on the hierarchical structure.

### Available Methods

- `.metrics().create_counter()`: Create a counter metric
- `.metrics().create_gauge()`: Create a gauge metric
- `.metrics().create_histogram()`: Create a histogram metric
- `.metrics().create_countervec()`: Create a counter with labels
- `.metrics().create_gaugevec()`: Create a gauge with labels
- `.metrics().create_histogramvec()`: Create a histogram with labels

### Creating Metrics

```rust
use dynamo_runtime::DistributedRuntime;

let runtime = DistributedRuntime::new()?;
let endpoint = runtime.namespace("my_namespace").component("my_component").endpoint("my_endpoint");

// Simple metrics
let requests_total = endpoint.metrics().create_counter(
    "requests_total",
    "Total requests",
    &[]
)?;

let active_connections = endpoint.metrics().create_gauge(
    "active_connections",
    "Active connections",
    &[]
)?;

let latency = endpoint.metrics().create_histogram(
    "latency_seconds",
    "Request latency",
    &[],
    Some(vec![0.001, 0.01, 0.1, 1.0, 10.0])
)?;
```

### Using Metrics

```rust
// Counters
requests_total.inc();

// Gauges
active_connections.set(42.0);
active_connections.inc();
active_connections.dec();

// Histograms
latency.observe(0.023);  // 23ms
```

### Vector Metrics with Labels

```rust
// Create vector metrics with label names
let requests_by_model = endpoint.metrics().create_countervec(
    "requests_by_model",
    "Requests by model",
    &["model_type", "model_size"],
    &[]
)?;

let memory_by_gpu = endpoint.metrics().create_gaugevec(
    "gpu_memory_bytes",
    "GPU memory by device",
    &["gpu_id", "memory_type"],
    &[]
)?;

// Use with specific label values
requests_by_model.with_label_values(&["llama", "7b"]).inc();
memory_by_gpu.with_label_values(&["0", "allocated"]).set(8192.0);
```

### Advanced Features

**Custom histogram buckets:**
```rust
let latency = endpoint.metrics().create_histogram(
    "latency_seconds",
    "Request latency",
    &[],
    Some(vec![0.001, 0.01, 0.1, 1.0, 10.0])
)?;
```

**Constant labels:**
```rust
let counter = endpoint.metrics().create_counter(
    "requests_total",
    "Total requests",
    &[("region", "us-west"), ("env", "prod")]
)?;
```

---

## Metrics API in Python

Python components can create and manage Prometheus metrics using the same metrics API through Python bindings.

### Available Methods

- `endpoint.metrics.create_counter()` / `create_intcounter()`: Create a counter metric
- `endpoint.metrics.create_gauge()` / `create_intgauge()`: Create a gauge metric
- `endpoint.metrics.create_histogram()`: Create a histogram metric
- `endpoint.metrics.create_countervec()` / `create_intcountervec()`: Create a counter with labels
- `endpoint.metrics.create_gaugevec()` / `create_intgaugevec()`: Create a gauge with labels
- `endpoint.metrics.create_histogramvec()`: Create a histogram with labels

All metrics are imported from `dynamo.prometheus_metrics`.

### Creating Metrics

```python
from dynamo.runtime import DistributedRuntime

drt = DistributedRuntime()
endpoint = drt.namespace("my_namespace").component("my_component").endpoint("my_endpoint")

# Simple metrics
requests_total = endpoint.metrics.create_intcounter(
    "requests_total",
    "Total requests"
)

active_connections = endpoint.metrics.create_intgauge(
    "active_connections",
    "Active connections"
)

latency = endpoint.metrics.create_histogram(
    "latency_seconds",
    "Request latency",
    buckets=[0.001, 0.01, 0.1, 1.0, 10.0]
)
```

### Using Metrics

```python
# Counters
requests_total.inc()
requests_total.inc_by(5)

# Gauges
active_connections.set(42)
active_connections.inc()
active_connections.dec()

# Histograms
latency.observe(0.023)  # 23ms
```

### Vector Metrics with Labels

```python
# Create vector metrics with label names
requests_by_model = endpoint.metrics.create_intcountervec(
    "requests_by_model",
    "Requests by model",
    ["model_type", "model_size"]
)

memory_by_gpu = endpoint.metrics.create_intgaugevec(
    "gpu_memory_bytes",
    "GPU memory by device",
    ["gpu_id", "memory_type"]
)

# Use with specific label values
requests_by_model.inc({"model_type": "llama", "model_size": "7b"})
memory_by_gpu.set(8192, {"gpu_id": "0", "memory_type": "allocated"})
```

### Advanced Features

**Constant labels:**
```python
counter = endpoint.metrics.create_intcounter(
    "requests_total",
    "Total requests",
    [("region", "us-west"), ("env", "prod")]
)
```

**Metric introspection:**
```python
print(counter.name())            # "my_namespace_my_component_my_endpoint_requests_total"
print(counter.const_labels())    # {"dynamo_namespace": "my_namespace", ...}
print(gauge_vec.variable_labels())  # ["model_type", "model_size"]
```

**Update patterns:**

Background thread updates:
```python
import threading
import time

def update_loop():
    while True:
        active_connections.set(compute_current_connections())
        time.sleep(2)

threading.Thread(target=update_loop, daemon=True).start()
```

Callback-based updates (called before each `/metrics` scrape):
```python
def update_metrics():
    active_connections.set(compute_current_connections())

endpoint.metrics.register_callback(update_metrics)
```

### Examples

Example scripts: [lib/bindings/python/examples/metrics/](../../lib/bindings/python/examples/metrics/)

```bash
cd ~/dynamo/lib/bindings/python/examples/metrics
DYN_SYSTEM_PORT=8081 ./server_with_loop.py
DYN_SYSTEM_PORT=8081 ./server_with_callback.py
```

---

## Related Documentation

- [Metrics Overview](metrics.md)
- [Prometheus and Grafana Setup](prometheus-grafana.md)
- [Distributed Runtime Architecture](../design_docs/distributed_runtime.md)
- [Python Metrics Examples](../../lib/bindings/python/examples/metrics/)

