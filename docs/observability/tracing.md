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

# Distributed Tracing with Tempo

This guide explains how to set up and view distributed traces in Grafana Tempo for Dynamo workloads.

## Overview

Dynamo supports OpenTelemetry-based distributed tracing, allowing you to visualize request flows across Frontend and Worker components. Traces are exported to Tempo via OTLP (OpenTelemetry Protocol) and visualized in Grafana.

## Prerequisites

- Docker and Docker Compose (for local deployment)
- Kubernetes cluster with kubectl access (for Kubernetes deployment)
- Dynamo runtime with tracing support

## Environment Variables

Dynamo's tracing is configured via environment variables. For complete logging documentation, see [logging.md](./logging.md).

### Required Environment Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `DYN_LOGGING_JSONL` | Enable JSONL logging format (required for tracing) | `true` |
| `OTEL_EXPORT_ENABLED` | Enable OTLP trace export | `1` |
| `OTEL_EXPORT_ENDPOINT` | OTLP gRPC endpoint for Tempo | `http://localhost:4317` (local) or `http://tempo:4317` (docker) |
| `OTEL_SERVICE_NAME` | Service name for identifying components | `dynamo-frontend`, `dynamo-worker-prefill`, `dynamo-worker-decode` |

### Example Configuration

```bash
# Enable JSONL logging and tracing
export DYN_LOGGING_JSONL=true

# Enable trace export to Tempo
export OTEL_EXPORT_ENABLED=1

# Set the Tempo endpoint (from host machine)
export OTEL_EXPORT_ENDPOINT=http://localhost:4317

# Set service name to identify this component
export OTEL_SERVICE_NAME=dynamo-frontend
```

---

## Local Deployment with Docker Compose

### 1. Start the Observability Stack

From the `deploy` directory, start Prometheus, Tempo, and Grafana:

```bash
cd deploy
docker compose --profile metrics up -d
```

This will start:
- **Prometheus** on `http://localhost:9090` for metrics
- **Tempo** on `http://localhost:3200` (HTTP API) and `localhost:4317` (OTLP gRPC) for traces
- **Grafana** on `http://localhost:3001` (username: `dynamo`, password: `dynamo`)

Verify services are running:

```bash
docker compose --profile metrics ps
```

### 2. Set Environment Variables

Configure Dynamo components to export traces:

```bash
# Enable JSONL logging and tracing
export DYN_LOGGING_JSONL=true
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORT_ENDPOINT=http://localhost:4317

# Set service names for each component
export OTEL_SERVICE_NAME=dynamo-frontend
```

### 3. Run a Dynamo Deployment

Many launch scripts support the `--enable-otel` flag to automatically configure tracing. For example:

```bash
# Navigate to SGLang launch directory
cd components/backends/sglang/launch

# Run aggregated deployment with tracing enabled
./agg.sh --enable-otel
```

Alternatively, you can manually set the environment variables before running any deployment script.

### 4. Generate Traces

Send requests to the frontend to generate traces:

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

### 5. View Traces in Grafana

1. Open Grafana at `http://localhost:3001`
2. Login with username `dynamo` and password `dynamo`
3. Navigate to **Explore** (compass icon in the left sidebar)
4. **Select "Tempo" as the datasource** from the dropdown at the top (it defaults to "Prometheus")
5. Use the **Search** tab to find traces:
   - Search by **Service Name** (e.g., `dynamo-frontend`)
   - Search by **Span Name** (e.g., `http-request`, `handle_payload`)
   - Search by **Tags** (e.g., `x_request_id=test-trace-001`)
6. Click on a trace to view the detailed flame graph

#### Filtering Traces

To see application-level spans and filter out HTTP overhead:

1. In the TraceQL tab, use:
   ```
   {span.name != "http-request"}
   ```

2. Or in the Search tab, set:
   - **Span Name** → **!=** → `http-request`

This will show you the `handle_payload` and other application spans.

#### Example Trace View

Below is an example of what a trace looks like in Grafana Tempo:

![Trace Example](../images/trace.png)

### 6. Stop Services

When done, stop the observability stack:

```bash
cd deploy
docker compose --profile metrics down
```

---

## Kubernetes Deployment

For Kubernetes deployments, ensure you have a Tempo instance deployed and accessible (e.g., `http://tempo.observability.svc.cluster.local:4317`).

### Modify DynamoGraphDeployment for Tracing

Add common tracing environment variables at the top level and service-specific names in each component in your `DynamoGraphDeployment`:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: sglang-disagg
spec:
  # Common environment variables for all services
  env:
    - name: DYN_LOGGING_JSONL
      value: "true"
    - name: OTEL_EXPORT_ENABLED
      value: "1"
    - name: OTEL_EXPORT_ENDPOINT
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

    SglangDecodeWorker:
      # ... existing configuration ...
      extraPodSpec:
        mainContainer:
          # ... existing configuration ...
          env:
            - name: OTEL_SERVICE_NAME
              value: "dynamo-worker-decode"

    SglangPrefillWorker:
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
kubectl apply -f your-deployment.yaml
```

Traces will now be exported to Tempo and can be viewed in Grafana.

---

## Unified Observability

The Docker Compose setup provides a unified observability stack:

- **Metrics**: View in Grafana Dashboards (datasource: Prometheus)
- **Traces**: View in Grafana Explore (datasource: Tempo)
- **Both accessible from**: `http://localhost:3001`

All observability data is collected in a single Grafana instance for seamless correlation between metrics and traces.

