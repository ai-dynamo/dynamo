# Adaptive Request Plane Client Guide

This guide explains how to use the new adaptive request plane client that automatically discovers services from etcd and adapts to their transport types without relying on the `DYN_REQUEST_PLANE` environment variable.

## Overview

The adaptive request plane client implements the following architecture:

- **Workers are explicit**: They know what transport they want to provide and register accordingly
- **Clients are adaptive**: They automatically adapt to whatever transports are available
- **No environment variables needed for clients**: Discovery is automatic via etcd
- **Single transport per component**: All instances of a particular component use the same transport type

## Key Components

### 1. AdaptiveRequestPlaneClient

The `AdaptiveRequestPlaneClient` is the core component that:
- Discovers services from etcd dynamically
- Inspects the transport field in the first discovered instance (all instances use same transport)
- Automatically creates the appropriate client (HTTP or NATS) based on what it finds
- Monitors instance changes and maintains transport client accordingly

### 2. Adaptive PushRouter

The `PushRouter` now has adaptive methods:
- `adaptive_from_client()` - Creates a router that discovers transports dynamically
- `adaptive_from_client_with_threshold()` - Same as above but with busy threshold support

## Usage Examples

### Example 1: Basic Adaptive Client

```rust
use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker,
    pipeline::network::egress::push_router::{PushRouter, RouterMode},
};

async fn adaptive_client_example(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Create client - this will discover services dynamically
    let client = drt
        .namespace("my-namespace")?
        .component("my-worker")?
        .client("my-endpoint")
        .await?;

    // Wait for instances to be discovered
    let instances = client.wait_for_instances().await?;
    tracing::info!("Discovered {} instances", instances.len());

    // Create adaptive push router - automatically uses discovered transports
    let router = PushRouter::<MyRequest, MyResponse>::adaptive_from_client(
        client,
        RouterMode::RoundRobin,
    ).await?;

    // Use the router normally
    let response = router.round_robin(request).await?;
    
    Ok(())
}
```

### Example 2: Worker Registration (Explicit Transport)

```rust
async fn worker_example(runtime: Runtime) -> Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Worker explicitly chooses its transport via environment variable
    // DYN_REQUEST_PLANE=http or DYN_REQUEST_PLANE=nats
    
    // Create and register service - transport info is automatically registered in etcd
    drt.namespace("my-namespace")?
        .component("my-worker")?
        .service_builder()
        .create()
        .await?
        .endpoint("my-endpoint")
        .endpoint_builder()
        .handler(my_ingress)
        .start()
        .await
}
```

## Migration Guide

### From Environment Variable Based Clients

**Before (relies on DYN_REQUEST_PLANE):**
```rust
// Client had to know the transport mode
let router = PushRouter::<Req, Resp>::from_client(client, RouterMode::RoundRobin).await?;
```

**After (adaptive discovery):**
```rust
// Client automatically discovers and adapts
let router = PushRouter::<Req, Resp>::adaptive_from_client(client, RouterMode::RoundRobin).await?;
```

### Worker Side (No Changes Required)

Workers continue to work as before - they explicitly set their transport mode:

```bash
# HTTP worker
DYN_REQUEST_PLANE=http cargo run --bin my-worker

# NATS worker  
DYN_REQUEST_PLANE=nats cargo run --bin my-worker
```

## Transport Discovery Flow

1. **Worker Startup**:
   - Worker reads `DYN_REQUEST_PLANE` environment variable
   - Registers service in etcd with transport information:
     - `TransportType::HttpTcp { http_endpoint }` for HTTP mode
     - `TransportType::NatsTcp(subject)` for NATS mode

2. **Client Startup**:
   - Client creates `ComponentClient` for service discovery
   - `AdaptiveRequestPlaneClient` monitors etcd for service instances
   - When first instance is discovered, detects component transport type:
     - `ComponentTransportType::Http` for HTTP transports
     - `ComponentTransportType::Nats` for NATS transports
   - Creates single appropriate transport client for the component

3. **Request Routing**:
   - Client automatically routes requests using the discovered transport
   - No environment variable needed on client side

## Benefits

1. **Simplified Deployment**: Clients don't need to know transport configuration
2. **Dynamic Adaptation**: Clients automatically adapt to component transport type
3. **Efficient Detection**: Only needs to check first instance (all instances use same transport)
4. **Consistent Components**: All instances of a component use the same transport type
5. **Deadlock Prevention**: Avoids deadlocks in bidirectional communication (A↔B scenarios)
6. **Backward Compatibility**: Existing workers continue to work unchanged

## Example Scenarios

### Scenario 1: HTTP-only Environment
- Worker: `DYN_REQUEST_PLANE=http` → registers HTTP endpoint in etcd
- Client: No env var → discovers HTTP endpoint → creates HTTP client
- Result: HTTP communication

### Scenario 2: NATS-only Environment  
- Worker: `DYN_REQUEST_PLANE=nats` → registers NATS subject in etcd
- Client: No env var → discovers NATS subject → creates NATS client
- Result: NATS communication

### Scenario 3: Multiple Components
- Component A (all instances): `DYN_REQUEST_PLANE=http` → all register HTTP endpoints
- Component B (all instances): `DYN_REQUEST_PLANE=nats` → all register NATS subjects  
- Client for A: No env var → discovers HTTP → creates HTTP client
- Client for B: No env var → discovers NATS → creates NATS client
- Result: Each component uses consistent transport, clients adapt accordingly

### Scenario 4: Bidirectional Communication (Deadlock Prevention)
- Component A needs to call Component B
- Component B needs to call Component A
- Both components use same transport type (e.g., HTTP)
- No deadlock risk: Both use the same transport protocol consistently
- Contrast with mixed transports: A(HTTP) ↔ B(NATS) could cause blocking issues

## Running the Demo

See the complete example in `lib/runtime/examples/adaptive_request_plane_demo.rs`:

```bash
# Start etcd
etcd

# Start HTTP worker
DYN_REQUEST_PLANE=http cargo run --example adaptive_request_plane_demo -- server

# Start adaptive client (no env var needed)
cargo run --example adaptive_request_plane_demo -- client
```

## API Reference

### AdaptiveRequestPlaneClient

- `new(component_client, nats_client)` - Create new adaptive client
- `send_request_to_instance(instance_id, payload, headers)` - Send to specific instance
- `get_transport_type()` - Get detected transport type for component
- `wait_for_transport_detection()` - Wait for transport type to be detected
- `is_http_transport()` - Check if component uses HTTP transport
- `is_nats_transport()` - Check if component uses NATS transport
- `get_instances()` - Get all available instances

### PushRouter Adaptive Methods

- `adaptive_from_client(client, router_mode)` - Create adaptive router
- `adaptive_from_client_with_threshold(client, router_mode, threshold, monitor)` - With busy detection

## Implementation Details

The adaptive client uses the existing service discovery infrastructure:
- Leverages `ComponentClient` for etcd watching
- Reuses existing transport implementations (`HttpRequestClient`, NATS client)
- Maintains compatibility with existing `RequestPlaneClient` trait
- Automatically handles instance lifecycle (add/remove/update)
- Assumes all instances of a component use the same transport type for efficiency
- Detects transport type from first discovered instance only

This ensures minimal changes to existing code while providing powerful adaptive capabilities with optimal performance.

## Deadlock Prevention in Bidirectional Communication

### The Problem
In distributed systems, deadlocks can occur when components A and B need to communicate with each other:
- Component A calls Component B
- Component B calls Component A (as part of processing A's request)
- If they use different transport protocols with different blocking characteristics, deadlocks can occur

### How Single Transport Per Component Helps

**Consistent Behavior**: When all instances of a component use the same transport type:
- All communication follows the same protocol semantics
- Connection pooling and resource management is consistent
- Timeout and retry behavior is predictable

**Example Deadlock Scenario (Avoided)**:
```
Component A (HTTP) → Component B (NATS)
Component B (NATS) → Component A (HTTP)

Potential issue: Different connection models and blocking behaviors
```

**With Single Transport (Safe)**:
```
Component A (HTTP) → Component B (HTTP)
Component B (HTTP) → Component A (HTTP)

Safe: Consistent connection model and behavior
```

### Implementation Notes

The adaptive client ensures deadlock prevention by:
1. **Transport Consistency**: All instances of a component use the same transport
2. **Early Detection**: Transport type is detected from the first instance
3. **Connection Reuse**: Same transport client can be reused for all instances
4. **Predictable Behavior**: No mixed transport protocols within a component

This design eliminates a whole class of potential deadlock scenarios in complex distributed workflows.
