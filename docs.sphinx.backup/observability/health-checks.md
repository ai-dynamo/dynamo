<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Health Checks

## Overview

Dynamo provides health check and liveness HTTP endpoints for each component which
can be used to configure startup, liveness and readiness probes in
orchestration frameworks such as Kubernetes.

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DYN_SYSTEM_PORT` | System status server port | `8081` | `9090` |
| `DYN_SYSTEM_STARTING_HEALTH_STATUS` | Initial health status | `notready` | `ready`, `notready` |
| `DYN_SYSTEM_HEALTH_PATH` | Custom health endpoint path | `/health` | `/custom/health` |
| `DYN_SYSTEM_LIVE_PATH` | Custom liveness endpoint path | `/live` | `/custom/live` |
| `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` | Endpoints required for ready state | none | `["generate"]` |
| `DYN_HEALTH_CHECK_ENABLED` | Enable canary health checks | `false` (K8s: `true`) | `true`, `false` |
| `DYN_CANARY_WAIT_TIME` | Seconds before sending canary health check | `10` | `5`, `30` |
| `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | Health check request timeout in seconds | `3` | `5`, `10` |

## Getting Started Quickly

Enable health checks and query endpoints:

```bash
# Start your Dynamo components (default port 8000, override with --http-port or DYN_HTTP_PORT env var)
python -m dynamo.frontend &

# Enable system status server on port 8081
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &
```

Check health status:

```bash
# Frontend health (port 8000)
curl -s localhost:8000/health | jq

# Worker health (port 8081)
curl -s localhost:8081/health | jq
```

## Frontend Liveness Check

The frontend liveness endpoint reports a status of `live` as long as
the service is running.

> **Note**: Frontend liveness doesn't depend on worker health or liveness only on the Frontend service itself.

### Example Request

```
curl -s localhost:8080/live -q | jq
```

### Example Response

```
{
  "message": "Service is live",
  "status": "live"
}
```

## Frontend Health Check

The frontend health endpoint reports a status of `healthy` as long as
the service is running.  Once workers have been registered, the
`health` endpoint will also list registered endpoints and instances.

> **Note**: Frontend liveness doesn't depend on worker health or liveness only on the Frontend service itself.

### Example Request

```
curl -v localhost:8080/health -q | jq
```

### Example Response

Before workers are registered:

```
HTTP/1.1 200 OK
content-type: application/json
content-length: 72
date: Wed, 03 Sep 2025 13:31:44 GMT

{
  "instances": [],
  "message": "No endpoints available",
  "status": "unhealthy"
}
```

After workers are registered:

```
HTTP/1.1 200 OK
content-type: application/json
content-length: 609
date: Wed, 03 Sep 2025 13:32:03 GMT

{
  "endpoints": [
    "dyn://dynamo.backend.generate"
  ],
  "instances": [
    {
      "component": "backend",
      "endpoint": "clear_kv_blocks",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.clear_kv_blocks-694d98147d54be25"
      }
    },
    {
      "component": "backend",
      "endpoint": "generate",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.generate-694d98147d54be25"
      }
    },
    {
      "component": "backend",
      "endpoint": "load_metrics",
      "instance_id": 7587888160958628000,
      "namespace": "dynamo",
      "transport": {
        "nats_tcp": "dynamo_backend.load_metrics-694d98147d54be25"
      }
    }
  ],
  "status": "healthy"
}
```

## Worker Liveness and Health Check

Health checks for components other than the frontend are enabled
selectively based on environment variables. If a health check for a
component is enabled the starting status can be set along with the set
of endpoints that are required to be served before the component is
declared `ready`.

Once all endpoints declared in `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS`
are served the component transitions to a `ready` state until the
component is shutdown. The endpoints return HTTP status code of `HTTP/1.1 503 Service Unavailable`
when initializing and HTTP status code `HTTP/1.1 200 OK` once ready.

> **Note**: Both /live and /ready return the same information

### Example Environment Setting

```
export DYN_SYSTEM_PORT=9090
export DYN_SYSTEM_STARTING_HEALTH_STATUS="notready"
export DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS="[\"generate\"]"
```

#### Example Request

```
curl -v localhost:9090/health | jq
```

#### Example Response
Before endpoints are being served:

```
HTTP/1.1 503 Service Unavailable
content-type: text/plain; charset=utf-8
content-length: 96
date: Wed, 03 Sep 2025 13:42:39 GMT

{
  "endpoints": {
    "generate": "notready"
  },
  "status": "notready",
  "uptime": {
    "nanos": 313803539,
    "secs": 12
  }
}
```

After endpoints are being served:

```
HTTP/1.1 200 OK
content-type: text/plain; charset=utf-8
content-length: 139
date: Wed, 03 Sep 2025 13:42:45 GMT

{
  "endpoints": {
    "clear_kv_blocks": "ready",
    "generate": "ready",
    "load_metrics": "ready"
  },
  "status": "ready",
  "uptime": {
    "nanos": 356504530,
    "secs": 18
  }
}
```

## Canary Health Checks (Active Monitoring)

In addition to the HTTP endpoints described above, Dynamo includes a **canary health check** system that actively monitors worker endpoints.

### Overview

The canary health check system:
- **Monitors endpoint health** by sending periodic test requests to worker endpoints
- **Only activates during idle periods** - if there's ongoing traffic, health checks are skipped to avoid overhead
- **Automatically enabled in Kubernetes** deployments via the operator
- **Disabled by default** in local/development environments

### How It Works

1. **Idle Detection**: After no activity on an endpoint for a configurable wait time (default: 10 seconds), a canary health check is triggered
2. **Health Check Request**: A lightweight test request is sent to the endpoint with a minimal payload (generates 1 token)
3. **Activity Resets Timer**: If normal requests arrive, the canary timer resets and no health check is sent
4. **Timeout Handling**: If a health check doesn't respond within the timeout (default: 3 seconds), the endpoint is marked as unhealthy

### Configuration

#### In Kubernetes (Enabled by Default)

Health checks are automatically enabled by the Dynamo operator. No additional configuration is required.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    VllmWorker:
      componentType: worker
      replicas: 2
      # Health checks automatically enabled by operator
```

#### In Local/Development Environments (Disabled by Default)

To enable health checks locally:

```bash
# Enable health checks
export DYN_HEALTH_CHECK_ENABLED=true

# Optional: Customize timing
export DYN_CANARY_WAIT_TIME=5  # Wait 5 seconds before sending health check
export DYN_HEALTH_CHECK_REQUEST_TIMEOUT=5  # 5 second timeout

# Start worker
python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

#### Configuration Options

| Environment Variable | Description | Default | Notes |
|---------------------|-------------|---------|-------|
| `DYN_HEALTH_CHECK_ENABLED` | Enable/disable canary health checks | `false` (K8s: `true`) | Automatically set to `true` in K8s |
| `DYN_CANARY_WAIT_TIME` | Seconds to wait (during idle) before sending health check | `10` | Lower values = more frequent checks |
| `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | Max seconds to wait for health check response | `3` | Higher values = more tolerance for slow responses |

### Health Check Payloads

Each backend defines its own minimal health check payload:

- **vLLM**: Single token generation with minimal sampling options
- **TensorRT-LLM**: Single token with BOS token ID
- **SGLang**: Single token generation request

These payloads are designed to:
- Complete quickly (< 100ms typically)
- Minimize GPU overhead
- Verify the full inference stack is working

### Observing Health Checks

When health checks are enabled, you'll see logs like:

```
INFO Health check manager started (canary_wait_time: 10s, request_timeout: 3s)
INFO Spawned health check task for endpoint: generate
INFO Canary timer expired for generate, sending health check
INFO Health check successful for generate
```

If an endpoint fails:

```
WARN Health check timeout for generate
ERROR Health check request failed for generate: connection refused
```

### When to Use Canary Health Checks

**Enable in production (Kubernetes):**
- ✅ Detect unhealthy workers before they affect user traffic
- ✅ Enable faster failure detection and recovery
- ✅ Monitor worker availability continuously

**Disable in development:**
- ✅ Reduce log noise during debugging
- ✅ Avoid overhead when not needed
- ✅ Simplify local testing

### Troubleshooting

**Health checks timing out:**
- Increase `DYN_HEALTH_CHECK_REQUEST_TIMEOUT`
- Check worker logs for errors
- Verify network connectivity

**Too many health check logs:**
- Increase `DYN_CANARY_WAIT_TIME` to reduce frequency
- Or disable with `DYN_HEALTH_CHECK_ENABLED=false` in dev

**Health checks not running:**
- Verify `DYN_HEALTH_CHECK_ENABLED=true` is set
- Check that `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` includes the endpoint
- Ensure the worker is serving the endpoint

## Related Documentation

- [Distributed Runtime Architecture](../design_docs/distributed_runtime.md)
- [Dynamo Architecture Overview](../design_docs/architecture.md)
- [Backend Guide](../development/backend-guide.md)
