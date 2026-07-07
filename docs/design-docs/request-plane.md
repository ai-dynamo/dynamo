---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Plane
---

## Overview

Dynamo supports two transport mechanisms for its request plane (the communication layer between services):

- **TCP** (default): Direct TCP connection for optimal performance
- **NATS**: Message broker-based request plane

This guide explains how to configure and use request plane in your Dynamo deployment.

## What is a Request Plane?

The request plane is the transport layer that handles communication between Dynamo services (e.g., frontend to backend, worker to worker). Different request planes offer different trade-offs:

| Request Plane | Suitable For | Characteristics |
|--------------|----------|-----------------|
| **NATS** | Deployments that choose brokered request transport | Requires NATS infrastructure, provides pub/sub patterns, highest flexibility |
| **TCP** | Low-latency direct communication | Direct connections, minimal overhead |

## Request Plane vs KV Event Plane

Dynamo has **two independent communication planes**:

- **Request plane** (**`DYN_REQUEST_PLANE`**): how **RPC requests** flow between components (frontend → router → worker), via `tcp`, or `nats`.
- **KV event plane** (**`DYN_EVENT_PLANE`**): how **KV cache events** (and optional router replica sync) are distributed for KV-aware routing, via `nats` or `zmq`.

**Note:** If you are using `tcp` request plane with KV events enabled on the router (the default router-side setting), the configured event plane is initialized independently. NATS-based event transport uses `NATS_SERVER` (default `nats://localhost:4222`), while ZMQ avoids external NATS infrastructure. SGLang requires explicit `--kv-events-config` and TRT-LLM requires `--publish-events-and-metrics` to publish events. For vLLM, KV events are currently auto-configured when prefix caching is active (deprecated — use `--kv-events-config` explicitly to prepare for a future release where all backends will default to off). To disable the router's KV event listener, use `--no-router-kv-events` on the frontend.

Because they are independent, you can mix them.

For example, a deployment with TCP request plane can use different KV event planes:
- **JetStream KV events**: requests use TCP, KV routing still uses NATS JetStream + object store for persistence.
- **NATS Core KV events (local indexer)**: requests use TCP, KV events use NATS Core pub/sub and persistence lives on workers.
- **no KV events**: requests use TCP and KV routing runs without events (no NATS required, but no event-backed persistence).

## Transport Security

The TCP request plane does not authenticate peers or encrypt connections. Keep its dynamically registered ports on
a trusted network and allow traffic only between the Dynamo components that require it.

The NATS request plane uses the same client authentication settings as the NATS event plane. Configure credentials,
TLS, and network access as described in [Event Plane Transport Security](event-plane.md#transport-security).

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

**Note:** By default, TCP uses an OS-assigned free port (port 0). This is ideal for environments where multiple services may run on the same machine or when you want to avoid port conflicts. If you need a specific port (e.g., for firewall rules), set `DYN_TCP_RPC_PORT` explicitly.

**When to use TCP:**
- Simple deployments with direct service-to-service communication (e.g. frontend to backend)
- Minimal infrastructure requirements (NATS is initialized when the router listens for KV events; disable with `--no-router-kv-events`)
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

### Using NATS

NATS provides durable jetstream messaging for request plane and can be used for KV events (and router replica sync).

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
- Production deployments with service discovery
- Event-backed KV-aware routing when using NATS as the event transport. Note: ZMQ event transport and approximate mode (`--no-router-kv-events`) both provide KV routing without NATS, with approximate mode using predicted cache state.
- Need for message replay and persistence features

Limitations:
- NATS does not support payloads beyond 16MB (use TCP for larger payloads)

## Complete Example

Here's a complete example showing how to launch a Dynamo deployment with different request planes:

See [`examples/backends/vllm/launch/agg_request_planes.sh`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/agg_request_planes.sh) for a complete working example that demonstrates launching Dynamo with TCP or NATS request planes.


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
