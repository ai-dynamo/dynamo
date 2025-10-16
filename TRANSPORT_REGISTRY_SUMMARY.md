# Transport Registry - Quick Reference

## Overview

This document provides a quick reference for the Transport Registry system in Dynamo. For detailed architectural information, see [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md).

## Current State

### Supported Transport Combinations

| Mode | Request Plane | Response Plane | Configuration | Status |
|------|---------------|----------------|---------------|--------|
| Legacy | NATS | TCP (call-home) | `DYN_REQUEST_PLANE=nats` | Default |
| Recommended | HTTP/2 | TCP (call-home) | `DYN_REQUEST_PLANE=http` | Production |

### Key Components

```
lib/runtime/src/pipeline/network/
├── request_plane.rs          # Request plane abstraction (existing)
├── transport_registry.rs     # New transport registry (this design)
├── tcp/
│   ├── client.rs            # TCP client for response streaming
│   └── server.rs            # TCP server for response streaming
├── egress/
│   └── http_router.rs       # HTTP request client
└── ingress/
    └── http_endpoint.rs     # HTTP request server
```

## Planned Enhancements

### Priority 0: TCP Connection Pooling (IMMEDIATE)

**Goal**: Eliminate 3-way handshake overhead for TCP response connections

**What Changes**:
- Add `TcpResponseConnectionPool` in `tcp/connection_pool.rs`
- Update `TcpClient` to reuse connections
- Update `TcpStreamServer` to support persistent connections

**Benefits**:
- **5-10% p50 latency reduction**
- **10-20% p99 latency reduction**
- Lower CPU usage
- Fewer file descriptors

**Configuration**:
```bash
# Enable connection pooling
export DYN_TCP_POOL_ENABLED=true

# Optional: Tune pool parameters
export DYN_TCP_POOL_MAX_PER_TARGET=50
export DYN_TCP_POOL_MIN_IDLE=5
export DYN_TCP_POOL_IDLE_TIMEOUT_SECS=300
```

**Impact**: Minimal code changes, backward compatible, can be feature-flagged

---

### Priority 1: HTTP Response Path (3-6 MONTHS)

**Goal**: Pure HTTP/2 mode (HTTP request + HTTP response)

**What Changes**:
- Add `HttpResponseServer` in `ingress/http_response_endpoint.rs`
- Add `HttpResponseClient` in `egress/http_response_client.rs`
- Support SSE (Server-Sent Events) streaming format
- Update service discovery to advertise HTTP response endpoints

**Benefits**:
- Simplified deployment (no call-home pattern)
- Standard HTTP tooling support
- Better proxy/load balancer compatibility
- Easier debugging

**Configuration**:
```bash
# Enable HTTP response path
export DYN_REQUEST_PLANE=http
export DYN_RESPONSE_PLANE=http

# Optional: Tune HTTP response server
export DYN_HTTP_RESPONSE_HOST=0.0.0.0
export DYN_HTTP_RESPONSE_PORT=8082
export DYN_HTTP_RESPONSE_STREAM_FORMAT=sse  # sse, ndjson, or binary
```

**Trade-offs**:
- Slightly higher latency than TCP (HTTP framing overhead)
- Different error model

---

### Priority 2: TCP Request Path (6-12 MONTHS)

**Goal**: Full TCP mode (TCP request + TCP response) for lowest latency

**What Changes**:
- Add `TcpRequestServer` in `ingress/tcp_request_endpoint.rs`
- Add `TcpRequestClient` in `egress/tcp_request_client.rs`
- Implement multiplexing protocol
- Persistent connections with heartbeat

**Benefits**:
- **Lowest possible latency**
- Single persistent connection
- Request/response multiplexing
- Bidirectional streaming support

**Configuration**:
```bash
# Enable TCP request path
export DYN_REQUEST_PLANE=tcp
export DYN_RESPONSE_PLANE=tcp

# TCP request server
export DYN_TCP_REQUEST_HOST=0.0.0.0
export DYN_TCP_REQUEST_PORT=9001
export DYN_TCP_REQUEST_MAX_CONNECTIONS=1000
```

**Trade-offs**:
- Custom protocol (harder to debug)
- Not compatible with HTTP-only proxies

---

## Service Discovery

### Current (Legacy)

```rust
pub enum TransportType {
    NatsTcp(String),
    HttpTcp { http_endpoint: String },
}

pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: i64,
    pub transport: TransportType,
}
```

### Future (Unified)

```rust
pub enum TransportType {
    // Legacy (backward compatible)
    NatsTcp(String),
    HttpTcp { http_endpoint: String },

    // New unified format
    Unified {
        request_transport: TransportEndpoint,
        response_transport: TransportEndpoint,
    },
}

pub struct TransportEndpoint {
    pub transport_id: TransportId,  // Http, Tcp, Nats, Grpc
    pub address: String,
    pub capabilities: TransportCapabilities,
    pub metadata: HashMap<String, String>,
}
```

## Using the Transport Registry

### Registering a Transport

```rust
use dynamo::pipeline::network::transport_registry::*;

// Create registry
let registry = Arc::new(TransportRegistry::new());

// Register HTTP request transport
let registration = TransportRegistration {
    transport_id: TransportId::Http,
    plane_type: PlaneType::Request,
    capabilities: TransportCapabilities {
        streaming: false,
        persistent_connections: true,
        bidirectional: false,
        max_message_size: None,
    },
    priority: 10,
};

let client = Arc::new(HttpRequestClient::new()?);
let server = Arc::new(HttpRequestServer::new(handler));

registry.register_request_transport(registration, client, server).await?;
```

### Selecting a Transport

```rust
// Automatically select best transport based on capabilities
let required = TransportCapabilities {
    streaming: true,
    persistent_connections: true,
    bidirectional: false,
    max_message_size: Some(10 * 1024 * 1024),  // 10MB
};

let transport_id = registry.select_transport(
    PlaneType::Response,
    &required,
).await?;

// Get client for selected transport
let client = registry.get_response_client(&transport_id).await
    .ok_or_else(|| anyhow!("Transport not available"))?;
```

### Creating Response Streams

```rust
// Create response connection info
let connection_info = ResponseConnectionInfo {
    address: TransportAddress::Http {
        url: "http://worker:8082/v1/rpc/response/my-subject".to_string(),
    },
    context_id: context.id().to_string(),
    subject: Some("my-subject".to_string()),
    metadata: HashMap::new(),
};

// Create response stream
let mut stream = client.create_response_stream(
    context.clone(),
    connection_info,
).await?;

// Send responses
stream.send(data).await?;
stream.send_control(ControlMessage::Stop).await?;
stream.close().await?;
```

## Migration Path

### Phase 0: Enable Connection Pooling (Week 1-2)

1. Set environment variable:
   ```bash
   export DYN_TCP_POOL_ENABLED=true
   ```

2. Monitor metrics:
   - `tcp_pool_connections_reused_total`
   - `tcp_pool_connections_created_total`
   - `tcp_pool_active_connections`

3. Validate latency improvements in p50/p95/p99

### Phase 1: Pilot HTTP Response Path (Month 1-3)

1. Deploy with HTTP response on test endpoints:
   ```bash
   export DYN_REQUEST_PLANE=http
   export DYN_RESPONSE_PLANE=http
   ```

2. Compare metrics with TCP baseline:
   - Request latency
   - Throughput
   - Error rates

3. Gradual rollout to production endpoints

### Phase 2: Pilot TCP Request Path (Month 6-12)

1. Deploy on internal high-throughput endpoints:
   ```bash
   export DYN_REQUEST_PLANE=tcp
   export DYN_RESPONSE_PLANE=tcp
   ```

2. Validate multiplexing and connection management

3. Measure performance vs HTTP baseline

## Environment Variables Reference

### Request Plane (Existing)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DYN_REQUEST_PLANE` | `nats`, `http` | `nats` | Request plane transport |
| `DYN_HTTP_RPC_HOST` | hostname | `0.0.0.0` | HTTP request server bind address |
| `DYN_HTTP_RPC_PORT` | port | `8081` | HTTP request server port |
| `DYN_HTTP_RPC_ROOT_PATH` | path | `/v1/rpc` | HTTP request API root path |
| `DYN_HTTP_REQUEST_TIMEOUT` | seconds | `5` | HTTP request timeout |

### Response Plane (New)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DYN_RESPONSE_PLANE` | `tcp`, `http` | `tcp` | Response plane transport |
| `DYN_HTTP_RESPONSE_HOST` | hostname | `0.0.0.0` | HTTP response server bind address |
| `DYN_HTTP_RESPONSE_PORT` | port | `8082` | HTTP response server port |
| `DYN_HTTP_RESPONSE_ROOT_PATH` | path | `/v1/rpc/response` | HTTP response API root path |
| `DYN_HTTP_RESPONSE_STREAM_FORMAT` | `sse`, `ndjson`, `binary` | `sse` | Response streaming format |
| `DYN_HTTP_RESPONSE_TIMEOUT_SECS` | seconds | `3600` | Response stream timeout |

### TCP Connection Pool (Priority 0)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DYN_TCP_POOL_ENABLED` | bool | `false` | Enable connection pooling |
| `DYN_TCP_POOL_MAX_PER_TARGET` | int | `50` | Max connections per target |
| `DYN_TCP_POOL_MIN_IDLE` | int | `5` | Min idle connections |
| `DYN_TCP_POOL_IDLE_TIMEOUT_SECS` | int | `300` | Idle connection timeout |
| `DYN_TCP_POOL_MAX_LIFETIME_SECS` | int | `3600` | Max connection lifetime |
| `DYN_TCP_POOL_KEEPALIVE_SECS` | int | `60` | TCP keepalive interval |

### TCP Request Path (Priority 2)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DYN_TCP_REQUEST_HOST` | hostname | `0.0.0.0` | TCP request server bind address |
| `DYN_TCP_REQUEST_PORT` | port | `9001` | TCP request server port |
| `DYN_TCP_REQUEST_MAX_CONNECTIONS` | int | `1000` | Max concurrent connections |
| `DYN_TCP_REQUEST_HEARTBEAT_INTERVAL_SECS` | int | `30` | Heartbeat interval |
| `DYN_TCP_REQUEST_CONNECTION_TIMEOUT_SECS` | int | `60` | Connection timeout |

## Metrics

### Connection Pool Metrics (Priority 0)

```
# Total connections created
tcp_pool_connections_created_total{target="host:port"}

# Total connections reused from pool
tcp_pool_connections_reused_total{target="host:port"}

# Current active connections
tcp_pool_active_connections{target="host:port"}

# Current idle connections
tcp_pool_idle_connections{target="host:port"}

# Connection acquisition latency
tcp_pool_acquisition_latency_seconds{target="host:port"}

# Connection validation failures
tcp_pool_validation_failures_total{target="host:port"}
```

### Transport Metrics

```
# Requests by transport type
transport_requests_total{transport="http|tcp|nats", plane="request|response"}

# Request latency by transport
transport_request_latency_seconds{transport="http|tcp|nats", plane="request|response"}

# Transport errors
transport_errors_total{transport="http|tcp|nats", error_type="timeout|connection|protocol"}

# Active streams
transport_active_streams{transport="http|tcp|nats", plane="request|response"}
```

## Architecture Diagrams

### Current: HTTP Request + TCP Response

```
┌─────────┐                                    ┌────────┐
│ Router  │                                    │ Worker │
│         │                                    │        │
│ ┌─────┐ │ ─────HTTP POST /subject─────────▶ │ ┌────┐ │
│ │HTTP │ │    (request + ConnectionInfo)     │ │HTTP│ │
│ │Client│◀─────────202 Accepted────────────── │ │Srv │ │
│ └─────┘ │                                    │ └────┘ │
│         │                                    │        │
│ ┌─────┐ │                                    │ ┌────┐ │
│ │TCP  │ │ ◀──────TCP connect (call-home)──── │ │TCP │ │
│ │Srv  │ │                                    │ │Cli │ │
│ │     │ │ ◀─────Response stream (TCP)─────── │ │    │ │
│ └─────┘ │                                    │ └────┘ │
└─────────┘                                    └────────┘
```

### Priority 0: With Connection Pool

```
┌─────────┐                                    ┌────────┐
│ Router  │                                    │ Worker │
│         │                                    │        │
│ ┌─────┐ │ ─────HTTP POST /subject─────────▶ │ ┌────┐ │
│ │HTTP │ │                                    │ │HTTP│ │
│ │Client│◀─────────202 Accepted────────────── │ │Srv │ │
│ └─────┘ │                                    │ └────┘ │
│         │                                    │        │
│ ┌─────┐ │    ╔════════════════════════╗     │ ┌────┐ │
│ │TCP  │ │ ═══║ Persistent Connection  ║═════▶│ │TCP │ │
│ │Srv  │ │    ║    (multiplexed)       ║◀═══  │ │Pool│ │
│ │     │ │ ◀══║  Response stream N     ║═════▶│ │    │ │
│ └─────┘ │    ╚════════════════════════╝     │ └────┘ │
└─────────┘                                    └────────┘
  Reused connection, no handshake overhead!
```

### Priority 1: HTTP Request + HTTP Response

```
┌─────────┐                                    ┌────────┐
│ Router  │                                    │ Worker │
│         │                                    │        │
│ ┌─────┐ │ ─────HTTP POST /subject─────────▶ │ ┌────┐ │
│ │HTTP │ │    (request)                      │ │HTTP│ │
│ │Client│◀─────200 OK + SSE stream────────── │ │Srv │ │
│ │     │ │    data: {...}                    │ │    │ │
│ │     │ │    data: {...}                    │ │    │ │
│ │     │ │    event: done                    │ │    │ │
│ └─────┘ │                                    │ └────┘ │
└─────────┘                                    └────────┘
  Simple, standard, HTTP/2 multiplexing!
```

### Priority 2: TCP Request + TCP Response

```
┌─────────┐                                    ┌────────┐
│ Router  │    ╔════════════════════════╗     │ Worker │
│         │    ║ Persistent TCP Conn    ║     │        │
│ ┌─────┐ │ ═══║                        ║═════▶│ ┌────┐ │
│ │TCP  │ │    ║ ┌──────────────────┐   ║     │ │TCP │ │
│ │Client│─────║─│ Request ID: A    │───║────▶│ │Srv │ │
│ │     │ │    ║ └──────────────────┘   ║     │ │    │ │
│ │     │◀────║─┬──────────────────┐───║──────│ │    │ │
│ │     │ │    ║ │ Response ID: A   │   ║     │ │    │ │
│ │     │ │    ║ │ (multiplexed)    │   ║     │ │    │ │
│ │     │◀────║─└──────────────────┘───║──────│ │    │ │
│ └─────┘ │    ╚════════════════════════╝     │ └────┘ │
└─────────┘                                    └────────┘
  Lowest latency, full control!
```

## Testing

### Unit Tests

```bash
# Test transport registry
cargo test --package dynamo-runtime --lib transport_registry

# Test connection pool
cargo test --package dynamo-runtime --lib tcp::connection_pool
```

### Integration Tests

```bash
# Test with different transport combinations
DYN_REQUEST_PLANE=http DYN_RESPONSE_PLANE=tcp cargo test --test transport_integration

# Test connection pool
DYN_TCP_POOL_ENABLED=true cargo test --test tcp_pool_integration
```

### Benchmarks

```bash
# Compare latency with and without connection pooling
cargo bench --bench tcp_pool_benchmark

# Compare different transport combinations
cargo bench --bench transport_comparison
```

## Troubleshooting

### Connection Pool Issues

**Problem**: Connections not being reused
```bash
# Check metrics
curl http://localhost:9090/metrics | grep tcp_pool_connections_reused

# Enable debug logging
export RUST_LOG=dynamo::pipeline::network::tcp::connection_pool=debug

# Check pool configuration
echo $DYN_TCP_POOL_ENABLED
echo $DYN_TCP_POOL_MAX_PER_TARGET
```

**Problem**: Connection pool exhaustion
```bash
# Increase pool size
export DYN_TCP_POOL_MAX_PER_TARGET=100

# Check active connections
curl http://localhost:9090/metrics | grep tcp_pool_active_connections

# Check for connection leaks
netstat -an | grep ESTABLISHED | wc -l
```

### HTTP Response Path Issues

**Problem**: SSE stream not working
```bash
# Test with curl
curl -N -H "Accept: text/event-stream" http://localhost:8082/v1/rpc/response/my-subject

# Enable HTTP debugging
export RUST_LOG=dynamo::pipeline::network::ingress::http_response=debug

# Try different stream format
export DYN_HTTP_RESPONSE_STREAM_FORMAT=ndjson
```

## FAQ

**Q: Should I use HTTP or TCP for response streaming?**

A: For most deployments, TCP with connection pooling (Priority 0) provides the best balance of performance and complexity. Use HTTP response path (Priority 1) if you need better proxy compatibility or simpler debugging. Use TCP request path (Priority 2) only for high-throughput internal services where latency is critical.

**Q: What's the performance impact of connection pooling?**

A: Expected 5-10% p50 latency reduction and 10-20% p99 reduction by eliminating TCP handshake overhead. Minimal CPU/memory overhead from pool management.

**Q: Can I mix transport types in the same deployment?**

A: Yes! The service discovery system allows different endpoints to use different transport combinations. This enables gradual migration and A/B testing.

**Q: How do I monitor transport health?**

A: Use the Prometheus metrics exposed at `/metrics`:
- `transport_requests_total`
- `transport_request_latency_seconds`
- `transport_errors_total`
- `tcp_pool_*` metrics

**Q: What happens during failover?**

A: Connection pools automatically handle connection failures by creating new connections. The transport registry allows fallback to alternative transports if configured.

## Next Steps

1. **Review Design**: Read [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md) for full details
2. **Implement Priority 0**: Start with TCP connection pooling for immediate benefits
3. **Enable Metrics**: Set up monitoring for transport and connection pool metrics
4. **Pilot Testing**: Test on non-critical endpoints first
5. **Gradual Rollout**: Use feature flags and gradual traffic shifting

## References

- [TRANSPORT_REGISTRY_DESIGN.md](TRANSPORT_REGISTRY_DESIGN.md) - Full architectural design
- [lib/runtime/src/pipeline/network/transport_registry.rs](lib/runtime/src/pipeline/network/transport_registry.rs) - Interface definitions
- [HTTP_REQUEST_PLANE_IMPLEMENTATION.md](HTTP_REQUEST_PLANE_IMPLEMENTATION.md) - Existing HTTP request plane docs
- [transport_agnostic_dynamo-v2.md](transport_agnostic_dynamo-v2.md) - Transport agnostic architecture

