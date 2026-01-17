# NIM Backend Metrics Mock Server

This directory contains a mock NIM (NVIDIA Inference Microservices) backend server for testing the frontend's on-demand metrics collection feature.

## Purpose

**NOTE: This is temporary code.** Once NIM starts using Dynamo backend components natively, this mock server and the associated NIM metrics polling code will be removed.

This example demonstrates:
- How the Dynamo frontend can poll external backends for metrics
- Dynamic metric generation and collection
- The `runtime_stats` endpoint pattern
- Integration between frontend metrics and backend services

## Running the Example

### 1. Start the Mock NIM Backend

**Static mode (default - NATS only, no etcd):**
```bash
python3 examples/custom_backend/nim/mock_nim_backend.py
```

**Dynamic mode (with etcd for service discovery):**
```bash
python3 examples/custom_backend/nim/mock_nim_backend.py
```

This starts a backend on `nim.backend.runtime_stats` (default) that returns incrementing metrics. You can customize with `--custom-backend-metrics-endpoint "namespace.component.endpoint"`.

### 2. Start the Frontend with Metrics Polling

```bash
python3 -m dynamo.frontend \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --custom-backend-metrics-endpoint nim.backend.runtime_stats \
    --custom-backend-metrics-polling-interval 9.2
```

**Note:** The custom backend metrics polling works in both static (NATS-only) and dynamic (with etcd) modes. The frontend automatically detects and adapts to the backend's mode.

### 3. Query Metrics

```bash
curl http://localhost:8000/metrics
```

The frontend will periodically (every 9.2 seconds in this example):
1. Poll the mock NIM backend via the `runtime_stats` endpoint
2. Parse the returned metrics
3. Update Prometheus gauges

When you query the `/metrics` endpoint, you'll see the most recently polled metrics.

## Metrics Exposed

The mock server returns:

**Gauges:**
- `kv_cache_usage_perc` - Cycles between 0.30 and 0.93

**Note:** All metrics collected from custom backends are automatically prefixed with `dynamo_component_` when exposed via the frontend's `/metrics` endpoint. For example, the gauge `kv_cache_usage_perc` from the backend will appear as `dynamo_component_kv_cache_usage_perc` in Prometheus metrics.

## Implementation Details

The frontend's NIM metrics collection is implemented in:
- `lib/llm/src/http/service/custom_backend_metrics.rs` - Custom backend metrics collection (temporary)
- `lib/llm/src/http/service/metrics.rs` - Metrics router
- `components/src/dynamo/frontend/main.py` - `--custom-backend-metrics-polling-interval` flag

All NIM-specific code is marked with TODO comments for removal once NIM adopts Dynamo backend.



