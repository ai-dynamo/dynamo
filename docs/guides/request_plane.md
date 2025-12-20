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

# Dynamo Request Planes User Guide

## Overview

Dynamo supports multiple transport mechanisms for its request plane (the communication layer between services). You can choose from three different request plane modes based on your deployment requirements:

- **NATS** (default): Message broker-based request plane
- **TCP**: Direct TCP connection for optimal performance
- **HTTP**: HTTP/2-based request plane

This guide explains how to configure and use request plane in your Dynamo deployment.

## What is a Request Plane?

The request plane is the transport layer that handles communication between Dynamo services (e.g., frontend to backend, worker to worker). Different request planes offer different trade-offs:

| Request Plane | Suitable For | Characteristics |
|--------------|----------|-----------------|
| **NATS** | Production deployments with KV routing | Requires NATS infrastructure, provides pub/sub patterns, highest flexibility |
| **TCP** | Low-latency direct communication | Direct connections, minimal overhead |
| **HTTP** | Standard deployments, debugging | HTTP/2 protocol, easier observability with standard tools, widely compatible |

## KV Routing and NATS

Dynamo's Key-Value (KV) cache based routing optimizes large language model inference by intelligently directing requests to workers with the most relevant KV cache data. KV-aware routing improves both Time To First Token (TTFT) through better cache locality and Inter-Token Latency (ITL) through intelligent load balancing.

Please refer to the [KV Cache Routing documentation](../router/kv_cache_routing.md) for more details.

There are two modes of KV based routing:
- Exact KV routing (needs NATS):  KV routing is based KV events indexing in a radix tree scoring the best match for the request. *This requires NATS* to persist and distribute KV events across routers.

- Approximate KV routing (does not need NATS): KV routing is based on approximate load heuristics. *This does not require NATS*.

## Configuration

### Environment Variable

Set the request plane mode using the `DYN_REQUEST_PLANE` environment variable:

```bash
export DYN_REQUEST_PLANE=<mode>
```

Where `<mode>` is one of:
- `nats` (default)
- `tcp`
- `http`

The value is case-insensitive.

### Default Behavior

If `DYN_REQUEST_PLANE` is not set or contains an invalid value, Dynamo defaults to `nats`.

## Usage Examples

### Using NATS (Default)

NATS is the default request plane and provides the most flexibility for complex deployments.

**Prerequisites:**
- NATS server must be running and accessible
- Configure NATS connection via standard Dynamo NATS environment variables

```bash
# Explicitly set to NATS (optional, as it's the default)

# Run your Dynamo service
DYN_REQUEST_PLANE=nats python -m dynamo.frontend --http-port=8000 &
DYN_REQUEST_PLANE=nats python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

**When to use NATS:**
- Production deployments with service discovery
- Currently (HA) highly available routers require durable messages persisted in NATS message broker. If you want to completely disable NATS, KV based routing won't be available
- Multiple frontends and backends
- Need for message replay and persistence features

Limitations:
- NATS does not support payloads beyond 16MB (use TCP for larger payloads)

### Using TCP

TCP provides direct, low-latency communication between services.

**Configuration:**

```bash
# Set request plane to TCP
export DYN_REQUEST_PLANE=tcp

# Optional: Configure TCP server host and port
export DYN_TCP_RPC_HOST=0.0.0.0  # Default host
export DYN_TCP_RPC_PORT=9999     # Default port

# Run your Dynamo service
DYN_REQUEST_PLANE=tcp python -m dynamo.frontend --http-port=8000 &
DYN_REQUEST_PLANE=tcp python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

**When to use TCP:**
- Simple deployments with direct service-to-service communication (e.g. frontend to backend)
- Minimal infrastructure requirements (no NATS needed)
- Low-latency requirements

**TCP Configuration Options:**

Additional TCP-specific environment variables:
- `DYN_TCP_RPC_HOST`: Server host address (default: auto-detected)
- `DYN_TCP_RPC_PORT`: Server port (default: 9999)
- `DYN_TCP_MAX_MESSAGE_SIZE`: Maximum message size for TCP client (default: 32MB)
- `DYN_TCP_REQUEST_TIMEOUT`: Request timeout for TCP client (default: 10 seconds)
- `DYN_TCP_POOL_SIZE`: Connection pool size for TCP client (default: 50)
- `DYN_TCP_CONNECT_TIMEOUT`: Connect timeout for TCP client (default: 3 seconds)
- `DYN_TCP_CHANNEL_BUFFER`: Request channel buffer size for TCP client (default: 100)

### Using HTTP

HTTP/2 provides a standards-based request plane that's easy to debug and widely compatible.

**Configuration:**

```bash
# Optional: Configure HTTP server host and port
export DYN_HTTP_RPC_HOST=0.0.0.0      # Default host
export DYN_HTTP_RPC_PORT=8888         # Default port
export DYN_HTTP_RPC_ROOT_PATH=/v1/rpc # Default path

# Run your Dynamo service
DYN_REQUEST_PLANE=http python -m dynamo.frontend --http-port=8000 &
DYN_REQUEST_PLANE=http python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

**When to use HTTP:**
- Standard deployments requiring HTTP compatibility
- Debugging scenarios (use curl, browser tools, etc.)
- Integration with HTTP-based infrastructure
- Load balancers and proxies that work with HTTP

**HTTP Configuration Options:**

Additional HTTP-specific environment variables:
- `DYN_HTTP_RPC_HOST`: Server host address (default: auto-detected)
- `DYN_HTTP_RPC_PORT`: Server port (default: 8888)
- `DYN_HTTP_RPC_ROOT_PATH`: Root path for RPC endpoints (default: /v1/rpc)

`DYN_HTTP2_*`: Various HTTP/2 client configuration options
- `DYN_HTTP2_MAX_FRAME_SIZE`: Maximum frame size for HTTP client (default: 1MB)
- `DYN_HTTP2_MAX_CONCURRENT_STREAMS`: Maximum concurrent streams for HTTP client (default: 1000)
- `DYN_HTTP2_POOL_MAX_IDLE_PER_HOST`: Maximum idle connections per host for HTTP client (default: 100)
- `DYN_HTTP2_POOL_IDLE_TIMEOUT_SECS`: Idle timeout for HTTP client (default: 90 seconds)
- `DYN_HTTP2_KEEP_ALIVE_INTERVAL_SECS`: Keep-alive interval for HTTP client (default: 30 seconds)
- `DYN_HTTP2_KEEP_ALIVE_TIMEOUT_SECS`: Keep-alive timeout for HTTP client (default: 10 seconds)
- `DYN_HTTP2_ADAPTIVE_WINDOW`: Enable adaptive flow control (default: true)

## Complete Example

Here's a complete example showing how to launch a Dynamo deployment with different request planes:

See [`examples/backends/vllm/launch/agg_request_planes.sh`](../../examples/backends/vllm/launch/agg_request_planes.sh) for a complete working example that demonstrates launching Dynamo with TCP, HTTP, or NATS request planes.


## Real-World Example

The Dynamo repository includes a complete example demonstrating all three request planes:

**Location:** `examples/backends/vllm/launch/agg_request_planes.sh`

```bash
cd examples/backends/vllm/launch

# Run with TCP
./agg_request_planes.sh --tcp

# Run with HTTP
./agg_request_planes.sh --http

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

1. **Mode Selection**: `DYN_REQUEST_PLANE` (defaults to `nats`)
2. **Transport-Specific Config**: Mode-specific environment variables (e.g., `DYN_TCP_*`, `DYN_HTTP2_*`)

## Migration Guide

### From NATS to TCP

1. Stop your Dynamo services
2. Set environment variable `DYN_REQUEST_PLANE=tcp`
3. Optionally configure TCP-specific settings (`DYN_TCP_RPC_PORT`, etc.)
4. Restart your services


### From NATS to HTTP

1. Stop your Dynamo services
2. Set environment variable `DYN_REQUEST_PLANE=http`
3. Optionally configure HTTP-specific settings (`DYN_HTTP_RPC_PORT`, etc.)
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
- For TCP/HTTP: Ensure host/port configurations are correct and accessible
- For NATS: Verify NATS server is running and accessible

### Issue: "Invalid request plane mode" Error

**Symptoms:** Service fails to start with configuration error

**Solutions:**
- Check `DYN_REQUEST_PLANE` spelling (valid values: `nats`, `tcp`, `http`)
- Value is case-insensitive but must be one of the three options
- If not set, defaults to `nats`

### Issue: Port Conflicts

**Symptoms:** Server fails to start due to "address already in use"

**Solutions:**
- TCP default port: 9999 (adjust environment variable `DYN_TCP_RPC_PORT`)
- HTTP default port: 8888 (adjust environment variable `DYN_HTTP_RPC_PORT`)

## Performance Considerations

### Latency

- **TCP**: Lowest latency due to direct connections and binary serialization
- **HTTP**: Moderate latency with HTTP/2 overhead
- **NATS**: Moderate latency due to nats jet stream persistence


### Resource Usage

- **TCP**: Minimal infrastructure (no additional services required)
- **HTTP**: Minimal infrastructure (no additional services required)
- **NATS**: Requires running NATS server (additional memory/CPU)
