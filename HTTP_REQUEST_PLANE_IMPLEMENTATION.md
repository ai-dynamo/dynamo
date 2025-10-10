# HTTP Request Plane Implementation

## Overview

This implementation replaces NATS with HTTP/2 for the request distribution plane while keeping the existing TCP response streaming unchanged. This provides a simpler, more standard architecture with better observability and easier deployment.

## Architecture

### Current (Hybrid) Architecture
- **Request Plane**: HTTP/2 POST (new) or NATS pub/sub (legacy)
- **Response Plane**: TCP streaming with call-home pattern (unchanged)

### Key Design Principles
1. **Backward Compatible**: NATS mode remains the default; HTTP is opt-in via `DYN_REQUEST_PLANE=http`
2. **Minimal Changes**: Existing TCP response streaming and handler logic unchanged
3. **Feature Parity**: HTTP mode provides same routing, tracing, and metrics as NATS
4. **Zero Dependencies on NATS**: When in HTTP mode, NATS is not required

## Implementation Details

### 1. Configuration (`lib/runtime/src/config/request_plane.rs`)

New `RequestPlaneMode` enum with environment variable support:

```rust
pub enum RequestPlaneMode {
    Nats,  // default
    Http,  // new
}

// Usage
let mode = RequestPlaneMode::from_env();  // reads DYN_REQUEST_PLANE
```

**Environment Variables:**
- `DYN_REQUEST_PLANE`: `nats` (default) or `http`
- `DYN_HTTP_RPC_HOST`: Bind address (default: `0.0.0.0`)
- `DYN_HTTP_RPC_PORT`: Bind port (default: `8081`)
- `DYN_HTTP_RPC_ROOT_PATH`: API path prefix (default: `/v1/dynamo`)
- `DYN_HTTP_REQUEST_TIMEOUT`: Timeout in seconds (default: `5`)

### 2. Request Plane Abstraction (`lib/runtime/src/pipeline/network/request_plane.rs`)

Defines transport-agnostic interfaces:

```rust
#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes>;
}

#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    async fn start(&mut self, bind_address: String) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn public_address(&self) -> String;
}
```

### 3. HTTP Server Endpoint (`lib/runtime/src/pipeline/network/ingress/http_endpoint.rs`)

Axum-based HTTP/2 server that:
- Accepts POST requests at `/v1/dynamo/{endpoint}`
- Extracts tracing headers (`traceparent`, `tracestate`, `x-request-id`, `x-dynamo-request-id`)
- Returns `202 Accepted` immediately (like NATS ack)
- Spawns async handler in background
- Supports graceful shutdown with inflight request tracking

**Key Features:**
- HTTP/2 with connection pooling
- Compression and tracing middleware via Tower
- Health status integration
- Metrics tracking (inflight, bytes, duration)

### 4. HTTP Client Router (`lib/runtime/src/pipeline/network/egress/http_router.rs`)

Reqwest-based HTTP/2 client that:
- Sends POST requests with two-part encoded payload
- Forwards tracing headers
- Uses HTTP/2 prior knowledge for efficiency
- Connection pooling (50 connections/host)
- Configurable timeouts

**Key Type:**
```rust
pub struct HttpRequestClient {
    client: reqwest::Client,  // HTTP/2
    request_timeout: Duration,
}
```

### 5. Updated AddressedPushRouter (`lib/runtime/src/pipeline/network/egress/addressed_router.rs`)

Now supports both transports via an enum:

```rust
enum RequestTransport {
    Nats(Client),
    Http(Arc<HttpRequestClient>),
}

impl AddressedPushRouter {
    pub fn new(nats_client, tcp_server) -> Result<Arc<Self>>  // NATS mode
    pub fn new_http(http_client, tcp_server) -> Result<Arc<Self>>  // HTTP mode
    pub fn from_mode(mode, nats_client, tcp_server) -> Result<Arc<Self>>  // Factory
}
```

The `generate` method branches on transport type:
- **NATS**: Publishes via `client.request_with_headers()`
- **HTTP**: POSTs via `http_client.send_request()`

Both paths:
- Register TCP response stream before sending request
- Package control message + request as TwoPartMessage
- Forward tracing headers
- Await TCP response stream provider (unchanged)

### 6. Service Discovery Updates

**TransportType enum** (`lib/runtime/src/component.rs`):
```rust
pub enum TransportType {
    NatsTcp(String),  // NATS subject
    HttpTcp {
        http_endpoint: String,  // e.g., http://worker:8081/v1/dynamo/...
        tcp_endpoint: String,   // e.g., worker:8081
    },
}
```

**Endpoint Registration** (`lib/runtime/src/component/endpoint.rs`):
- Conditionally creates NATS service endpoint or HTTP server based on `DYN_REQUEST_PLANE`
- Registers appropriate transport type in etcd
- HTTP mode binds Axum server on configured address
- NATS mode creates NATS service endpoint (existing behavior)

### 7. Message Flow

#### HTTP Request Flow:
```
[Client/Router]
  ├─ Register TCP response stream → get ConnectionInfo
  ├─ Encode TwoPartMessage(control_header + request)
  ├─ POST http://worker:8081/v1/dynamo/{subject}
  │   Headers: traceparent, tracestate, x-request-id, x-dynamo-request-id
  │   Body: TwoPartCodec-encoded bytes
  └─ Receive 202 Accepted

[Worker]
  ├─ HTTP handler receives POST
  ├─ Extract tracing headers
  ├─ Return 202 immediately
  ├─ Spawn: handle_payload(body_bytes)
  ├─ Decode TwoPartMessage → control + request
  ├─ Extract ConnectionInfo from control
  ├─ TCP call-home to client
  ├─ Stream responses over TCP
  └─ Send completion marker
```

#### Response Flow (unchanged):
```
[Worker] → TCP connect → [Client TCP Server]
  ├─ Send prologue (success/error)
  ├─ Stream response chunks
  └─ Send completion marker { complete_final: true }
```

### 8. Testing

**Unit Tests:**
- `request_plane.rs`: Mode parsing and environment variable handling
- `http_endpoint.rs`: Axum handler, tracing header extraction, graceful shutdown
- `http_router.rs`: HTTP client creation, timeout configuration, error handling

**Example:**
- `examples/http_request_plane_demo.rs`: End-to-end demo with echo service

### 9. Metrics and Observability

Both modes maintain identical metrics:
- `work_handler_requests_total`: Total requests processed
- `work_handler_request_duration_seconds`: Processing time histogram
- `work_handler_inflight_requests`: Current inflight gauge
- `work_handler_request_bytes_total`: Request bytes counter
- `work_handler_response_bytes_total`: Response bytes counter
- `work_handler_errors_total{error_type}`: Error counter by type

HTTP mode adds standard HTTP observability:
- Access logs via Tower `TraceLayer`
- Status codes (202 for ack, 4xx/5xx for errors)
- Standard HTTP metrics via Axum middleware

Distributed tracing headers are preserved end-to-end in both modes.

## Migration Guide

### For Users

**No changes required** - NATS remains the default:
```bash
# Current behavior (NATS)
cargo run

# Opt into HTTP mode
DYN_REQUEST_PLANE=http cargo run
```

**HTTP Mode Configuration:**
```bash
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_HOST=0.0.0.0
export DYN_HTTP_RPC_PORT=8081
export DYN_HTTP_RPC_ROOT_PATH=/v1/dynamo
```

### For Developers

**No code changes required** for existing components. The transport selection is automatic based on `DYN_REQUEST_PLANE`.

If you need to explicitly control the transport:
```rust
use dynamo_runtime::config::RequestPlaneMode;

let mode = RequestPlaneMode::from_env();
match mode {
    RequestPlaneMode::Nats => { /* NATS-specific logic */ }
    RequestPlaneMode::Http => { /* HTTP-specific logic */ }
}
```

## Deployment

### HTTP Mode (Recommended)
**Requirements:**
- etcd for service discovery

**Advantages:**
- No NATS cluster to manage
- Standard HTTP/2 load balancers (nginx, envoy)
- Better observability (HTTP metrics, access logs)
- Simpler deployment (one less service)

### NATS Mode (Legacy)
**Requirements:**
- etcd for service discovery
- NATS with JetStream

**Use When:**
- Migrating from existing NATS-based deployments
- Need NATS-specific features (JetStream persistence, etc.)

## Performance Considerations

**HTTP/2 Advantages:**
- Multiplexing: multiple requests over single connection
- Header compression (HPACK)
- Binary protocol
- Connection pooling (50 conns/host)

**Expected Overhead:**
- HTTP has ~5-10% higher latency vs raw NATS for ack
- Response streaming (TCP) performance is identical
- HTTP/2 multiplexing reduces connection overhead at scale

## Future Enhancements

### Planned (not included in this PR)
1. **HTTP Response Streaming**: Replace TCP with Server-Sent Events (SSE) or WebSockets
2. **Authentication**: mTLS, JWT, or API key auth on HTTP endpoints
3. **Rate Limiting**: Standard HTTP rate limiting middleware
4. **gRPC Support**: Alternative to HTTP/2 JSON for typed interfaces
5. **Load Balancing**: Native support for Envoy/nginx with HTTP endpoints

### Out of Scope
- KV Router: Still uses NATS/JetStream (separate module)
- Object Store: Still uses NATS object store (separate module)

## Files Changed

### New Files
- `lib/runtime/src/config/request_plane.rs` - Config enum
- `lib/runtime/src/pipeline/network/request_plane.rs` - Transport traits
- `lib/runtime/src/pipeline/network/ingress/http_endpoint.rs` - HTTP server
- `lib/runtime/src/pipeline/network/egress/http_router.rs` - HTTP client
- `lib/runtime/examples/http_request_plane_demo.rs` - Example
- `HTTP_REQUEST_PLANE_IMPLEMENTATION.md` - This document

### Modified Files
- `lib/runtime/Cargo.toml` - Added reqwest to dependencies
- `lib/runtime/src/config.rs` - Export RequestPlaneMode
- `lib/runtime/src/pipeline/network.rs` - Export request_plane module
- `lib/runtime/src/pipeline/network/ingress.rs` - Export http_endpoint
- `lib/runtime/src/pipeline/network/egress.rs` - Export http_router
- `lib/runtime/src/pipeline/network/egress/addressed_router.rs` - Support both transports
- `lib/runtime/src/pipeline/network/egress/push_router.rs` - Use mode-based factory
- `lib/runtime/src/component.rs` - Add HttpTcp transport type
- `lib/runtime/src/component/endpoint.rs` - Conditional endpoint creation
- `README.md` - Updated documentation

## Testing

### Run Unit Tests
```bash
cd lib/runtime
cargo test config::request_plane
cargo test network::ingress::http_endpoint
cargo test network::egress::http_router
```

### Run Example
```bash
# Start etcd
etcd

# Run HTTP mode example
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo
```

### Integration Test
```bash
# Terminal 1: Start etcd
etcd

# Terminal 2: Run worker with HTTP mode
DYN_REQUEST_PLANE=http cargo run --example server

# Terminal 3: Run client
cargo run --example client
```

## Rollback Plan

If issues arise, rollback is immediate:
```bash
# Revert to NATS mode
export DYN_REQUEST_PLANE=nats

# Or unset (defaults to NATS)
unset DYN_REQUEST_PLANE
```

No code changes or redeployment required.

## Questions?

- **Why keep TCP for responses?**: Existing call-home pattern works well; HTTP streaming (SSE/WS) can be added later incrementally.
- **Why not gRPC?**: HTTP/2 JSON is simpler, more debuggable, and easier to proxy. gRPC can be added as alternative later.
- **Performance impact?**: HTTP/2 multiplexing offsets overhead; response streaming (bulk of data) remains on TCP.
- **Security?**: mTLS, auth middleware, and network policies apply naturally to HTTP endpoints.

## Acknowledgments

This implementation leverages existing Dynamo infrastructure:
- TCP streaming server and codec (no changes)
- etcd service discovery (extended for HTTP addresses)
- Tracing and metrics framework (reused)
- Graceful shutdown and health checking (integrated)

