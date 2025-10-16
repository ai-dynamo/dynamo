# Dynamo runtime: Transport Agnostic Dynamo Pipelines
Status: Implemented

Authors: [biswapanda](https://github.com/biswapanda)

Category: Architecture

Reviewers: [Ryan Olson](https://github.com/ryanolson), [Neelay Shah](https://github.com/nnshah1), [Graham King](https://github.com/grahamking), [Maksim Khadkevich](https://github.com/hutm), [Rudy Pei](https://github.com/PeaBrane), [Kyle kranen](https://github.com/kkranen)


## Problem statement

Currently NATS is used by Dynamo for various purposes including request routing, event distribution, metrics publishing, and object storage. However, customers don't want to deploy or maintain NATS infrastructure, preferring simpler HTTP-based solutions. This creates deployment complexity and operational overhead.

**Key Issues:**
- NATS deployment and maintenance overhead for customers
- Complex infrastructure requirements (etcd + NATS)
- Customers prefer HTTP-based solutions for simplicity
- Need for transport flexibility without breaking existing functionality

## Solution Overview

This proposal implements a **transport-agnostic architecture** that supports both NATS and HTTP modes, allowing customers to choose their preferred transport mechanism. The solution provides:

1. **HTTP/2 Request Plane**: Direct HTTP communication between routers and workers
2. **Conditional NATS Usage**: NATS features only enabled when explicitly configured
3. **Backward Compatibility**: Existing NATS deployments continue to work unchanged
4. **Simplified Deployment**: HTTP mode requires only etcd for service discovery

## Implementation Approach

### Request Plane

**HTTP Mode (New - Recommended)**
- Uses HTTP/2 for request distribution between routers and workers
- Direct point-to-point communication via Axum/HTTP server
- Only requires etcd for service discovery
- Configurable via `DYN_REQUEST_PLANE=http`

**NATS Mode (Legacy - Default)**
- Maintains existing NATS-based request routing
- Full backward compatibility
- Default behavior when `DYN_REQUEST_PLANE=nats` or unset

**Transport Abstraction**
```rust
// Request plane client trait for sending requests
#[async_trait]
pub trait RequestPlaneClient: Send + Sync {
    async fn send_request(&self, address: String, payload: Bytes, headers: Headers) -> Result<Bytes>;
}

// Request plane server trait for receiving requests
#[async_trait]
pub trait RequestPlaneServer: Send + Sync {
    async fn start(&mut self, bind_address: String) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn public_address(&self) -> String;
}
```

### Event Plane

**HTTP Mode Behavior**
- KV router events are **not published to NATS** when in HTTP mode
- Events are consumed locally but not distributed (dummy event processor)
- Reduces NATS traffic and eliminates unnecessary network overhead
- KV metrics remain available via HTTP endpoint for polling

**NATS Mode Behavior**
- Full KV event broadcasting to all routers via NATS
- Maintains existing event subscription and publishing functionality
- Frontend can subscribe to KV router events as before

**Conditional Event Publishing**
```rust
// Only start NATS event publishing if we're using NATS request plane mode
let request_plane_mode = RequestPlaneMode::from_env();
if request_plane_mode.is_nats() {
    self.start_nats_metrics_publishing(namespace, worker_id);
} else {
    tracing::debug!("Skipping NATS metrics publishing - request plane mode is '{}'", request_plane_mode);
}
```

### Metrics and Monitoring

**HTTP Mode**
- **Disables NATS service stats collection** - no periodic `$SRV.STATS.*` requests
- **Disables NATS KV metrics publishing** - no `namespace.dynamo.kv_metrics` publications
- **Maintains Prometheus metrics** - all application metrics still available
- **KV metrics endpoint available** - can be polled via HTTP instead of NATS subscription

**NATS Mode**
- Full NATS metrics publishing and service stats collection
- Maintains existing monitoring and observability features

### Object Store

**Current Implementation**
- NATS object store continues to be used for model files and router snapshots
- No changes to existing object store functionality in this phase

**Future Alternatives** (Not implemented in this phase)
1. **Direct Model Pulling**: Frontend pulls model files directly from HuggingFace/ModelExpress
2. **HTTP Model Serving**: Backend instances serve model files over HTTP interface
3. **Cloud Object Storage**: Integration with S3/GCS/Azure Blob storage

## Configuration

### Environment Variables

**Request Plane Mode**
```bash
# Use HTTP/2 for request distribution (recommended for new deployments)
export DYN_REQUEST_PLANE=http

# Use NATS for request distribution (default, maintains backward compatibility)
export DYN_REQUEST_PLANE=nats
```

**HTTP Server Configuration**
```bash
# HTTP RPC server bind address (default: 0.0.0.0:8081)
export DYN_HTTP_RPC_HOST=0.0.0.0
export DYN_HTTP_RPC_PORT=8081

# HTTP RPC root path (default: /v1/rpc)
export DYN_HTTP_RPC_ROOT_PATH=/v1/rpc

# HTTP request timeout in seconds (default: 5)
export DYN_HTTP_REQUEST_TIMEOUT=5
```

### Infrastructure Requirements

**HTTP Mode**
- ✅ **etcd only** - for service discovery
- ❌ **No NATS required** - eliminates NATS deployment and maintenance

**NATS Mode**
- ✅ **etcd** - for service discovery
- ✅ **NATS with JetStream** - for request routing and events

## Migration Guide

### From NATS to HTTP Mode

**1. Update Environment Configuration**
```bash
# Before (NATS mode)
export DYN_REQUEST_PLANE=nats  # or unset

# After (HTTP mode)
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_HOST=0.0.0.0
export DYN_HTTP_RPC_PORT=8081
```

**2. Infrastructure Changes**
- **Keep etcd running** - still required for service discovery
- **NATS can be removed** - no longer needed for request routing
- **Update firewall rules** - ensure HTTP RPC port (8081) is accessible

**3. Monitoring Considerations**

**If Using KV Metrics Subscriber**
The KV metrics subscriber in HTTP mode will no longer receive push-based updates from NATS. Options:

1. **Poll the KV metrics endpoint**: Use existing `KV_METRICS_ENDPOINT` to periodically query
2. **Use Prometheus**: Scrape Prometheus metrics instead of NATS events
3. **Keep NATS mode**: If push-based KV metrics are required, use `DYN_REQUEST_PLANE=nats`

**Example: Polling KV Metrics Endpoint**
```rust
// In HTTP mode, poll the KV metrics endpoint instead of subscribing to NATS
let client = /* your HTTP client or router */;
let mut interval = tokio::time::interval(Duration::from_millis(100));

loop {
    interval.tick().await;

    // Make request to KV_METRICS_ENDPOINT
    let response = router.call(/* KV metrics request */).await?;

    // Process metrics
    let metrics: ForwardPassMetrics = /* deserialize response */;
    handle_metrics(metrics);
}
```

## Benefits and Trade-offs

### Benefits of HTTP Mode

**Operational Simplicity**
- **Reduced Infrastructure**: Only etcd required, no NATS deployment/maintenance
- **Simpler Architecture**: Fewer moving parts and dependencies
- **Standard Protocols**: HTTP/2 is widely understood and supported
- **Resource Efficiency**: Eliminates NATS overhead and background tasks

**Performance**
- **Reduced NATS Load**: No periodic stats requests or KV metrics publications
- **Direct Communication**: Point-to-point HTTP requests without NATS intermediary
- **Lower Latency**: Eliminates NATS message routing overhead

**Monitoring and Debugging**
- **Cleaner Logs**: No NATS trace logs cluttering monitoring
- **Standard HTTP Tools**: Can use curl, HTTP load balancers, etc.
- **Familiar Debugging**: Standard HTTP status codes and error handling

### Trade-offs

**Event Distribution**
- **No KV Event Broadcasting**: KV events not distributed across routers in HTTP mode
- **Polling Required**: KV metrics must be polled instead of push-based updates
- **Local Event Processing**: Events processed locally but not shared

**Backward Compatibility**
- **NATS Mode Preserved**: Existing deployments continue to work unchanged
- **Feature Parity**: All core functionality available in both modes
- **Gradual Migration**: Can migrate incrementally without breaking changes

## Implementation Details

### Core Components

**1. HTTP Request Client** (`lib/runtime/src/pipeline/network/egress/http_request_client.rs`)
```rust
pub struct HttpRequestClient {
    client: reqwest::Client,
    request_timeout: Duration,
}

impl HttpRequestClient {
    pub async fn send_request(&self, url: String, payload: Bytes, headers: HashMap<String, String>) -> Result<Bytes> {
        let mut request = self.client.post(&url).body(payload);

        for (key, value) in headers {
            request = request.header(&key, &value);
        }

        let response = request.timeout(self.request_timeout).send().await?;
        Ok(response.bytes().await?)
    }
}
```

**2. HTTP Endpoint Server** (`lib/runtime/src/pipeline/network/ingress/http_endpoint.rs`)
```rust
pub struct HttpEndpoint {
    pub service_handler: Arc<dyn PushWorkHandler>,
    pub cancellation_token: CancellationToken,
    pub graceful_shutdown: bool,
}

// Axum HTTP handler
async fn handle_request(AxumState(state): AxumState<HttpEndpointState>, body: Bytes) -> impl IntoResponse {
    // Spawn async handler for request processing
    tokio::spawn(async move {
        let result = service_handler.handle_payload(body).await;
        // Handle result and update metrics
    });

    // Return 202 Accepted immediately (like NATS ack)
    (StatusCode::ACCEPTED, "")
}
```

**3. Request Plane Mode Configuration** (`lib/runtime/src/config/request_plane.rs`)
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum RequestPlaneMode {
    Nats,
    Http,
}

impl RequestPlaneMode {
    pub fn from_env() -> Self {
        match std::env::var("DYN_REQUEST_PLANE").as_deref() {
            Ok("http") => RequestPlaneMode::Http,
            Ok("nats") | Err(_) => RequestPlaneMode::Nats, // Default to NATS
        }
    }

    pub fn is_nats(&self) -> bool { matches!(self, RequestPlaneMode::Nats) }
    pub fn is_http(&self) -> bool { matches!(self, RequestPlaneMode::Http) }
}
```

### Service Discovery Integration

**HTTP Mode Service Registration**
```rust
// Register HTTP endpoint in etcd for service discovery
let http_endpoint = format!("http://{}:{}{}/{}",
    host, port, root_path, subject);

// Store in etcd under generate endpoint
etcd_client.put(
    format!("{}/{:x}", endpoint.etcd_root(), instance_id),
    serde_json::to_string(&InstanceInfo {
        transport: TransportType::HttpTcp {
            http_endpoint,
        },
        instance_id,
        // ... other fields
    })?
).await?;
```

**Router Instance Discovery**
```rust
// Router discovers HTTP endpoints from etcd
let instances = etcd_client.get_prefix(endpoint.etcd_root()).await?;
for (_, value) in instances {
    let instance: InstanceInfo = serde_json::from_str(&value)?;
    match instance.transport {
        TransportType::HttpTcp { http_endpoint, .. } => {
            // Use HTTP endpoint for requests
            self.http_client.send_request(http_endpoint, payload, headers).await?;
        }
        TransportType::NatsTcp(subject) => {
            // Use NATS subject for requests
            self.nats_client.publish(subject, payload).await?;
        }
    }
}
```

### Conditional NATS Features

**Service Stats Collection** (`lib/runtime/src/component/service.rs`)
```rust
// Only collect NATS service stats in NATS mode
let request_plane_mode = RequestPlaneMode::from_env();
if request_plane_mode.is_nats() {
    component.start_scraping_nats_service_component_metrics()?;
} else {
    tracing::debug!("Skipping NATS service metrics collection for '{}' - request plane mode is '{}'",
        component_name, request_plane_mode);
}
```

**KV Metrics Publishing** (`lib/llm/src/kv_router/publisher.rs`)
```rust
// Only publish KV metrics to NATS in NATS mode
let request_plane_mode = RequestPlaneMode::from_env();
if request_plane_mode.is_nats() {
    self.start_nats_metrics_publishing(namespace, worker_id);
} else {
    tracing::debug!("Skipping NATS metrics publishing for KV metrics - request plane mode is '{}' (worker_id: {})",
        request_plane_mode, worker_id);
}
```

**KV Router Background Task** (`lib/llm/src/kv_router/subscriber.rs`)
```rust
// Skip NATS connections in HTTP mode, use minimal etcd-only background task
let request_plane_mode = RequestPlaneMode::from_env();
if request_plane_mode.is_http() {
    tracing::debug!("Skipping KV router NATS background task - request plane mode is '{}' (component: {})",
        request_plane_mode, component.subject());

    return start_http_mode_background(component, remove_worker_tx, cancellation_token).await;
}
```

## Testing and Validation

### HTTP Request Plane Demo

A complete example is provided in `lib/runtime/examples/http_request_plane_demo.rs`:

```bash
# 1. Start etcd
etcd

# 2. Run the server with HTTP mode
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server

# 3. Run the client
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client

# 4. Test with curl
curl -X POST http://localhost:8081/v1/rpc/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello HTTP mode!"}'
```

### Verification Scripts

**Test NATS Traffic Elimination** (`test_no_nats_http_mode.sh`)
```bash
#!/bin/bash
# Verify that NATS connections are completely eliminated in HTTP mode

export DYN_REQUEST_PLANE=http
# Start your services and verify:
# - No NATS PING/PONG messages in NATS server logs
# - Log messages show "Skipping NATS" operations
# - Only etcd operations for service discovery
```

**Test KV Metrics Modes** (`test_kv_metrics_mode.sh`)
```bash
#!/bin/bash
# Test KV metrics behavior in different modes

# NATS mode - metrics published to NATS
export DYN_REQUEST_PLANE=nats

# HTTP mode - metrics available via HTTP endpoint only
export DYN_REQUEST_PLANE=http
```

## Debugging and Monitoring

### Check Current Mode
```bash
echo $DYN_REQUEST_PLANE
# Output: http (or nats, or empty for default)
```

### Verify NATS Metrics Disabled
Look for these log messages when starting services in HTTP mode:
```
[DEBUG] Skipping NATS service metrics collection for 'dynamo_backend' - request plane mode is 'http'
[DEBUG] Skipping NATS metrics publishing for KV metrics - request plane mode is 'http' (worker_id: 123)
[DEBUG] Skipping KV router NATS background task - request plane mode is 'http' (component: frontend)
```

### NATS Server Logs
In HTTP mode, NATS server logs should show **no traffic at all**:
```bash
# No NATS traffic - completely eliminated
# No PUB $SRV.STATS.* messages
# No PUB namespace.dynamo.kv_metrics messages
# No SUB _INBOX.* subscriptions
```

## Future Enhancements

### Object Store Abstraction
- Implement pluggable object store interface
- Support for cloud storage providers (S3, GCS, Azure Blob)
- Direct model file serving via HTTP endpoints

### Enhanced HTTP Features
- HTTP/3 support for improved performance
- gRPC alternative for typed interfaces
- Load balancing and circuit breaker patterns

### Event Plane Evolution
- HTTP-based event streaming (Server-Sent Events, WebSockets)
- Event sourcing and replay capabilities
- Distributed event bus alternatives

## Conclusion

This implementation successfully provides a **transport-agnostic architecture** that eliminates NATS dependency for customers preferring HTTP-based solutions while maintaining full backward compatibility.

**Key Achievements:**
- ✅ **HTTP/2 request plane** implemented with Axum/HTTP server
- ✅ **Conditional NATS usage** - only enabled when explicitly configured
- ✅ **Zero breaking changes** - existing NATS deployments work unchanged
- ✅ **Simplified deployment** - HTTP mode requires only etcd
- ✅ **Comprehensive testing** - examples and validation scripts provided

The solution addresses customer concerns about NATS operational complexity while preserving the flexibility to use NATS when its advanced features are needed. This provides a clear migration path for customers to adopt simpler HTTP-based deployments without sacrificing functionality.
