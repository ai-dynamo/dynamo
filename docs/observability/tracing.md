<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Distributed Tracing with Tempo

This guide explains how to set up and view distributed traces in Grafana Tempo for Dynamo workloads.

> **Note:** This guide covers local single-instance deployments using Docker Compose for demonstration purposes. For distributed Kubernetes deployments, see the [Kubernetes Deployment](#kubernetes-deployment) section.

> **💡 Tip:** For local development, use the unified observability stack at `../../deploy/docker-observability.yml`, which includes Tempo, Prometheus, Grafana, and metric exporters in one convenient stack.

## Overview

Dynamo supports OpenTelemetry-based distributed tracing, allowing you to visualize request flows across Frontend and Worker components. Traces are exported to Tempo via OTLP (OpenTelemetry Protocol) and visualized in Grafana.

## Prerequisites

- Docker and Docker Compose (for local deployment)
- Kubernetes cluster with kubectl access (for Kubernetes deployment)
- Dynamo runtime with tracing support

## Environment Variables

Dynamo's tracing is configured via environment variables. For complete logging documentation, see [logging.md](./logging.md).

### Required Environment Variables

| Variable | Description | Default | Example Value |
|----------|-------------|---------|---------------|
| `DYN_LOGGING_JSONL` | Enable JSONL logging format (required for tracing) | `false` | `true` |
| `DYN_LOG` | Log level (info or debug). Use debug to see detailed traces. | `info` | `debug` |
| `OTEL_EXPORT_ENABLED` | Enable OTLP trace export | disabled | `1` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | OTLP gRPC endpoint for Tempo | `http://localhost:4317` | `http://tempo:4317` (docker) |
| `OTEL_SERVICE_NAME` | Service name for identifying components | `dynamo` | `dynamo-frontend`, `dynamo-worker-prefill`, `dynamo-worker-decode` |

### Tracing Behavior

- **When `OTEL_EXPORT_ENABLED` is NOT set**: Tracing is disabled. Traces are generated locally for trace logging, but not exported to any backend.
- **When `OTEL_EXPORT_ENABLED=1`**: Traces are exported via OTLP to the endpoint specified by `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
  - If `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is NOT set, it defaults to `http://localhost:4317`
  - If `OTEL_SERVICE_NAME` is NOT set, it defaults to `dynamo`

**Note:** When `OTEL_EXPORT_ENABLED=1`, logging initialization is deferred until the runtime is available (required by the OTEL exporter). This means some early logs will be dropped. This will be fixed in a future release.

### Log Levels and Visibility

Traces are exported to Tempo/Grafana regardless of log level. Set `DYN_LOG` only if you want to view trace details in console logs for debugging.

JSONL logs are written to **stderr**:

- **INFO (default)**: Initialization, errors, high-level operations
- **DEBUG**: Detailed traces with trace IDs, span IDs, and routing decisions

Example:
```bash
export DYN_LOGGING_JSONL=true
export DYN_LOG=debug
python -m dynamo.frontend --http-port 8000
```

### Example Configuration

```bash
# Enable JSONL logging and tracing
export DYN_LOGGING_JSONL=true

# Enable trace export to Tempo
export OTEL_EXPORT_ENABLED=1

# Set the Tempo endpoint (docker-compose network). Note that if this is not specified, it will default to http://localhost:4317
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo-host-name-here:4317

# Set service name to identify this component
export OTEL_SERVICE_NAME=dynamo-frontend
```

---

## Local Deployment with Docker Compose

> **Note:** The following Docker Compose commands are for demonstrating single-instance local deployments, not distributed Kubernetes environments.

### 1. Start the Unified Observability Stack

From the `deploy` directory, start the unified observability stack:

```bash
cd deploy
docker compose -f docker-observability.yml up -d
```

This will start:
- **Tempo** on `http://localhost:3200` (HTTP API) and `localhost:4317` (OTLP gRPC)
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3000` (username: `dynamo`, password: `dynamo`)
- **DCGM Exporter** on `http://localhost:9401/metrics` (GPU metrics)
- **NATS Exporter** on `http://localhost:7777/metrics`

Verify services are running:

```bash
docker compose -f docker-observability.yml ps
```

### 2. Set Environment Variables

Configure Dynamo components to export traces:

```bash
# Enable JSONL logging and tracing
export DYN_LOGGING_JSONL=true
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
```

### 3a. Quick Start: Single GPU Aggregated Deployment

For a simple single-GPU deployment, start the frontend and a single vLLM worker:

```bash
# Start the frontend with tracing enabled
export OTEL_SERVICE_NAME=dynamo-frontend
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# Start a single vLLM worker (aggregated prefill and decode)
export OTEL_SERVICE_NAME=dynamo-worker-vllm
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &

wait
```

This runs both prefill and decode on the same GPU, providing a simpler setup for testing tracing.

### 3b. Run vLLM Disaggregated Deployment (2 GPUs)

Run the vLLM disaggregated script with tracing enabled:

```bash
# Navigate to vLLM launch directory
cd examples/backends/vllm/launch

# Run disaggregated deployment (modify the script to export env vars first)
./disagg.sh
```

**Note:** You may need to modify `disagg.sh` to export the tracing environment variables before starting each component:

```bash
#!/bin/bash
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Enable tracing
export DYN_LOGGING_JSONL=true
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317

# Run frontend
export OTEL_SERVICE_NAME=dynamo-frontend
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# Run decode worker
export OTEL_SERVICE_NAME=dynamo-worker-decode
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &

# Run prefill worker
export OTEL_SERVICE_NAME=dynamo-worker-prefill
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker &

wait
```

For disaggregated deployments, this separates prefill and decode onto different GPUs for better resource utilization.

### 4. Generate Traces

Send requests to the frontend to generate traces (works for both aggregated and disaggregated deployments):

```bash
curl -d '{
  "model": "Qwen/Qwen3-0.6B",
  "max_completion_tokens": 100,
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}' \
-H 'Content-Type: application/json' \
-H 'x-request-id: test-trace-001' \
http://localhost:8000/v1/chat/completions
```

### 6. View Traces in Grafana Tempo

1. Open Grafana at `http://localhost:3000`
2. Login with username `dynamo` and password `dynamo`
3. Navigate to **Explore** (compass icon in the left sidebar)
4. Select **Tempo** as the data source (should be selected by default)
5. In the query type, select **"Search"** (not TraceQL, not Service Graph)
6. Use the **Search** tab to find traces:
   - Search by **Service Name** (e.g., `dynamo-frontend`)
   - Search by **Span Name** (e.g., `http-request`, `handle_payload`)
   - Search by **Tags** (e.g., `x_request_id=test-trace-001`)
7. Click on a trace to view the detailed flame graph

#### Example Trace View

Below is an example of what a trace looks like in Grafana Tempo:

![Trace Example](trace.png)

### 7. Stop Services

When done, stop the observability stack:

```bash
cd deploy
docker compose -f docker-observability.yml down
```

---

## Kubernetes Deployment

For Kubernetes deployments, ensure you have a Tempo instance deployed and accessible (e.g., `http://tempo.observability.svc.cluster.local:4317`).

### Modify DynamoGraphDeployment for Tracing

Add common tracing environment variables at the top level and service-specific names in each component in your `DynamoGraphDeployment` (e.g., `examples/backends/vllm/deploy/disagg.yaml`):

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  # Common environment variables for all services
  env:
    - name: DYN_LOGGING_JSONL
      value: "true"
    - name: OTEL_EXPORT_ENABLED
      value: "1"
    - name: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
      value: "http://tempo.observability.svc.cluster.local:4317"

  services:
    Frontend:
      # ... existing configuration ...
      extraPodSpec:
        mainContainer:
          # ... existing configuration ...
          env:
            - name: OTEL_SERVICE_NAME
              value: "dynamo-frontend"

    VllmDecodeWorker:
      # ... existing configuration ...
      extraPodSpec:
        mainContainer:
          # ... existing configuration ...
          env:
            - name: OTEL_SERVICE_NAME
              value: "dynamo-worker-decode"

    VllmPrefillWorker:
      # ... existing configuration ...
      extraPodSpec:
        mainContainer:
          # ... existing configuration ...
          env:
            - name: OTEL_SERVICE_NAME
              value: "dynamo-worker-prefill"
```

Apply the updated DynamoGraphDeployment:

```bash
kubectl apply -f examples/backends/vllm/deploy/disagg.yaml
```

Traces will now be exported to Tempo and can be viewed in Grafana.

