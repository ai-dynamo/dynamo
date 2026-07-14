---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Plane
---

## Overview

Dynamo supports two transport mechanisms for its request plane (the communication layer between services):

- **TCP** (default): Direct TCP connection for optimal performance
- **NATS** (legacy): Message broker-based request plane for existing deployments

This guide explains how to configure and use request plane in your Dynamo deployment.

## What is a Request Plane?

The request plane is the transport layer that handles communication between Dynamo services (e.g., frontend to backend, worker to worker). Different request planes offer different trade-offs:

| Request Plane | Suitable For | Characteristics |
|--------------|----------|-----------------|
| **NATS** | Existing deployments that require the legacy brokered transport | Requires NATS infrastructure and uses JetStream queues |
| **TCP** | Low-latency direct communication | Direct connections, minimal overhead |

> [!WARNING]
> The NATS request plane is legacy. Use the default TCP request plane for new deployments. NATS Core
> remains supported as an independent event-plane transport.

## Request Plane vs KV Event Plane

Dynamo has **two independent communication planes**:

- **Request plane** (**`DYN_REQUEST_PLANE`**): how **RPC requests** flow between components (frontend → router → worker), via `tcp`, or `nats`.
- **KV event plane** (**`DYN_EVENT_PLANE`**): how **KV cache events** (and optional router replica sync) are distributed for KV-aware routing, via `nats` or `zmq`.

> [!NOTE]
> The request and event planes are configured independently. ZMQ is the default event plane for
> every discovery backend. Set `DYN_EVENT_PLANE=nats` to opt into NATS Core event transport. See the
> [Router Quick Start](../components/router/README.md#quick-start) for backend KV event publication
> arguments.

Because they are independent, you can mix them.

For example, a deployment with TCP request plane can use different KV event planes:
- **ZMQ KV events (default)**: requests use TCP, and KV events use direct ZMQ pub/sub.
- **NATS Core KV events (local indexer)**: requests use TCP, KV events use NATS Core pub/sub and persistence lives on workers.
- **JetStream KV events (deprecated)**: requests use TCP, while KV routing uses NATS JetStream and object-store persistence.
- **No KV events**: requests use TCP and KV routing predicts cache state from routing decisions.

## Configuration

### Environment Variable

Set the request plane mode using the `DYN_REQUEST_PLANE` environment variable:

```bash
export DYN_REQUEST_PLANE=<mode>
```

Where `<mode>` is one of:
- `tcp` (default)
- `nats`

The value is case-insensitive.

### Default Behavior

If `DYN_REQUEST_PLANE` is not set or contains an invalid value, Dynamo defaults to `tcp`.

## Usage Examples

### Using TCP (Default)

TCP is the default request plane and provides direct, low-latency communication between services.

**Configuration:**

```bash
# TCP is the default, so no need to set DYN_REQUEST_PLANE explicitly
# But you can explicitly set it if desired:
export DYN_REQUEST_PLANE=tcp

# Optional: Configure TCP server host and port
export DYN_TCP_RPC_HOST=0.0.0.0  # Default host
# export DYN_TCP_RPC_PORT=9999   # Optional: specify a fixed port

# Run your Dynamo service
DYN_REQUEST_PLANE=tcp python -m dynamo.frontend --http-port=8000 &
DYN_REQUEST_PLANE=tcp python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

> [!NOTE]
> By default, TCP uses an OS-assigned free port (port 0). If firewall rules require a fixed port, set
> `DYN_TCP_RPC_PORT` explicitly.

**When to use TCP:**
- Simple deployments with direct service-to-service communication (e.g. frontend to backend)
- Minimal infrastructure requirements with the default ZMQ event plane
- Low-latency requirements

**TCP Configuration Options:**

Additional TCP-specific environment variables:
- `DYN_TCP_RPC_HOST`: Server host address (default: auto-detected)
- `DYN_TCP_RPC_PORT`: Server port. If not set, the OS assigns a free port automatically (recommended for most deployments). Set explicitly only if you need a specific port for firewall rules.
- `DYN_TCP_MAX_MESSAGE_SIZE`: Maximum message size for TCP client (default: 32MB)
- `DYN_TCP_SHRINK_MESSAGE_SIZE`: Threshold for shrinking the zero-copy decoder buffer back to initial size after processing large messages (default: 8MB, max: DYN_TCP_MAX_MESSAGE_SIZE)
- `DYN_TCP_REQUEST_TIMEOUT`: Request timeout for TCP client (default: 10 seconds)
- `DYN_TCP_POOL_SIZE`: Connection pool size for TCP client (default: 50)
- `DYN_TCP_CONNECT_TIMEOUT`: Connect timeout for TCP client (default: 3 seconds)
- `DYN_TCP_CHANNEL_BUFFER`: Request channel buffer size for TCP client (default: 100)

### Using NATS (Legacy)

NATS provides durable JetStream queues for the legacy request plane. This setting does not select
the event-plane transport.

**Prerequisites:**
- NATS server must be running and accessible
- Configure NATS connection via standard Dynamo NATS environment variables

```bash
# Explicitly set to NATS
export DYN_REQUEST_PLANE=nats

# Run your Dynamo service
DYN_REQUEST_PLANE=nats python -m dynamo.frontend --http-port=8000 &
DYN_REQUEST_PLANE=nats python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

**When to use NATS:**
- Existing deployments that already depend on the NATS request plane
- Workloads that require its brokered request queue and cannot yet migrate to TCP

Limitations:
- NATS does not support payloads beyond 16MB (use TCP for larger payloads)

## Complete Example

Here's a complete example showing how to launch a Dynamo deployment with different request planes:

See the [request-plane launch example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_request_planes.sh)
for a complete configuration that demonstrates launching Dynamo with TCP or NATS request planes.

## Real-World Example

The Dynamo repository includes a complete example demonstrating both request planes:

**Location:** `examples/backends/vllm/launch/agg_request_planes.sh`

```bash
cd examples/backends/vllm/launch

# Run with TCP
./agg_request_planes.sh --tcp

# Run with NATS
./agg_request_planes.sh --nats
```

## Architecture Details

### Network Manager

The request plane implementation is centralized in the Network Manager (`lib/runtime/src/pipeline/network/manager.rs`), which:

1. Reads the `DYN_REQUEST_PLANE` environment variable at startup
2. Creates the appropriate server and client implementations
3. Provides a transport-agnostic interface to the rest of the codebase
4. Manages all network configuration and lifecycle

### Transport Abstraction

All request plane implementations conform to common trait interfaces:
- `RequestPlaneServer`: Server-side interface for receiving requests
- `RequestPlaneClient`: Client-side interface for sending requests

This abstraction means your application code doesn't need to change when switching request planes.

### Configuration Loading

Request plane configuration is loaded from environment variables at startup and cached globally. The configuration hierarchy is:

1. **Mode Selection**: `DYN_REQUEST_PLANE` (defaults to `tcp`)
2. **Transport-Specific Config**: Mode-specific environment variables (e.g., `DYN_TCP_*`)

## Migration Guide

### From NATS to TCP

1. Stop your Dynamo services
2. Set environment variable `DYN_REQUEST_PLANE=tcp`
3. Optionally configure TCP-specific settings (e.g., `DYN_TCP_RPC_HOST`). Note: `DYN_TCP_RPC_PORT` is optional; if not set, an OS-assigned free port is used automatically.
4. Restart your services


### Testing the Migration

After switching request planes, verify your deployment:

```bash
# Test with a simple request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Troubleshooting

### Issue: Services Can't Communicate

**Symptoms:** Requests timeout or fail to reach the backend

**Solutions:**
- Verify all services use the same `DYN_REQUEST_PLANE` setting
- Check that server ports are not blocked by k8s network policies or firewalls
- For TCP: Ensure host/port configurations are correct and accessible
- For NATS: Verify NATS server is running and accessible

### Issue: "Invalid request plane mode" Error

**Symptoms:** Service fails to start with configuration error

**Solutions:**
- Check `DYN_REQUEST_PLANE` spelling (valid values: `nats`, `tcp`)
- Value is case-insensitive but must be one of the two options
- If not set, defaults to `tcp`

### Issue: Port Conflicts

**Symptoms:** Server fails to start due to "address already in use"

**Solutions:**
- TCP: By default, TCP uses an OS-assigned free port, so port conflicts should be rare. If you explicitly set `DYN_TCP_RPC_PORT` to a specific port and get conflicts, either change the port or remove the setting to use automatic port assignment.

## Performance Considerations

### Latency

- **TCP**: Lowest latency due to direct connections and binary serialization
- **NATS**: Moderate latency due to nats jet stream persistence


### Resource Usage

- **TCP**: Minimal request-plane infrastructure. KV events use the configured event plane; NATS is needed only when `DYN_EVENT_PLANE=nats`, and router-side event consumption can be disabled with `--no-router-kv-events`.
- **NATS**: Requires running NATS server (additional memory/CPU)
