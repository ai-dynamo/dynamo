---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Event Plane Architecture

This document describes Dynamo's event plane architecture, which handles service discovery, coordination, and event distribution using etcd and NATS.

## Overview

Dynamo's coordination layer adapts to the deployment environment:

| Deployment | Service Discovery | KV Events | Request Plane |
|------------|-------------------|-----------|---------------|
| **Kubernetes** (with operator) | Native K8s (CRDs, EndpointSlices) | NATS (optional) | TCP |
| **Bare metal / Local** (default) | etcd | NATS (optional) | TCP |

> **Note:** The runtime always defaults to `kv_store` (etcd) for service discovery. Kubernetes deployments must explicitly set `DYN_DISCOVERY_BACKEND=kubernetes` - the Dynamo operator handles this automatically.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Coordination Layer                                │
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Service Discovery     │    │            NATS                 │ │
│  │                         │    │         (Optional)              │ │
│  │  • K8s: CRDs + API      │    │  • KV Cache Events              │ │
│  │  • Bare metal: etcd     │    │  • Router Replica Sync          │ │
│  │                         │    │  • JetStream Persistence        │ │
│  └─────────────────────────┘    └─────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                    │                          │
         ┌──────────┴──────────┐    ┌─────────┴──────────┐
         ▼                     ▼    ▼                    ▼
    ┌─────────┐          ┌─────────┐              ┌─────────┐
    │Frontend │          │ Planner │              │ Worker  │
    └─────────┘          └─────────┘              └─────────┘
```

## Kubernetes-Native Service Discovery

When running on Kubernetes with the Dynamo operator, service discovery uses native Kubernetes resources instead of etcd.

### Configuration

The operator explicitly sets:
```bash
DYN_DISCOVERY_BACKEND=kubernetes
```

> **Important:** This must be explicitly configured. The runtime defaults to `kv_store` in all environments.

### How It Works

1. **DynamoWorkerMetadata CRD**: Workers register their endpoints by creating/updating DynamoWorkerMetadata custom resources
2. **EndpointSlices**: Used to signal readiness status to the system
3. **K8s API Watches**: Components watch for CRD changes to discover available endpoints

### Benefits

- No external etcd cluster required
- Native integration with Kubernetes lifecycle
- Automatic cleanup when pods terminate
- Works with standard K8s RBAC

### Environment Variables (Injected by Operator)

| Variable | Description |
|----------|-------------|
| `DYN_DISCOVERY_BACKEND` | Set to `kubernetes` |
| `POD_NAME` | Current pod name |
| `POD_NAMESPACE` | Current namespace |
| `POD_UID` | Pod unique identifier |

---

## etcd Architecture (Default for All Deployments)

When `DYN_DISCOVERY_BACKEND=kv_store` (the global default), etcd is used for service discovery.

### Connection Configuration

etcd connection is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ETCD_ENDPOINTS` | Comma-separated etcd URLs | `http://localhost:2379` |
| `ETCD_AUTH_USERNAME` | Basic auth username | None |
| `ETCD_AUTH_PASSWORD` | Basic auth password | None |
| `ETCD_AUTH_CA` | CA certificate path (TLS) | None |
| `ETCD_AUTH_CLIENT_CERT` | Client certificate path | None |
| `ETCD_AUTH_CLIENT_KEY` | Client key path | None |

Example:
```bash
export ETCD_ENDPOINTS=http://etcd-0:2379,http://etcd-1:2379,http://etcd-2:2379
```

### Lease Management

Each `DistributedRuntime` maintains a primary lease with etcd:

```
┌────────────────────┐         ┌──────────────┐
│ DistributedRuntime │◄────────│ Primary Lease │
│                    │         │  TTL: 10s     │
│  • Namespace       │         └───────┬───────┘
│  • Components      │                 │
│  • Endpoints       │                 │ Keep-Alive
│                    │                 │ Heartbeat
└────────────────────┘                 ▼
                               ┌──────────────┐
                               │     etcd     │
                               └──────────────┘
```

**Lease Lifecycle:**

1. **Creation**: Lease created during `DistributedRuntime` initialization
2. **Keep-Alive**: Background task sends heartbeats at 50% of remaining TTL
3. **Expiration**: If heartbeats stop, lease expires after TTL (10 seconds default)
4. **Cleanup**: All keys associated with the lease are automatically deleted

**Automatic Recovery:**

- Reconnection with exponential backoff (50ms to 5s)
- Deadline-based retry logic
- Cancellation token propagation

### Service Discovery

Endpoints are registered in etcd for dynamic discovery:

**Key Format:**
```
/services/{namespace}/{component}/{endpoint}/{instance_id}
```

**Example:**
```
/services/vllm-agg/backend/generate/694d98147d54be25
```

**Registration Data:**
```json
{
  "namespace": "vllm-agg",
  "component": "backend",
  "endpoint": "generate",
  "instance_id": 7587888160958628000,
  "transport": {
    "tcp": "192.168.1.10:9999"
  }
}
```

### Discovery Queries

The discovery system supports multiple query patterns:

| Query Type | Pattern | Use Case |
|------------|---------|----------|
| `AllEndpoints` | `/services/` | List all services |
| `NamespacedEndpoints` | `/services/{namespace}/` | Filter by namespace |
| `ComponentEndpoints` | `/services/{namespace}/{component}/` | Filter by component |
| `Endpoint` | `/services/{namespace}/{component}/{endpoint}/` | Specific endpoint |

### Watch Functionality

Clients watch etcd prefixes for real-time updates:

```python
# Client watches for endpoint changes
watcher = etcd.watch_prefix("/services/vllm-agg/backend/generate/")

for event in watcher:
    if event.type == "PUT":
        # New endpoint registered
        add_endpoint(event.value)
    elif event.type == "DELETE":
        # Endpoint removed (worker died)
        remove_endpoint(event.key)
```

**Watch Features:**

- Initial state retrieval with `get_and_watch_prefix()`
- Automatic reconnection on stream failure
- Revision tracking for no-event-loss guarantees
- Event types: `PUT` (create/update) and `DELETE`

### Distributed Locks

etcd provides distributed locking for coordination:

**Lock Types:**

| Type | Key Pattern | Behavior |
|------|-------------|----------|
| Write Lock | `v1/{prefix}/writer` | Exclusive (no readers/writers) |
| Read Lock | `v1/{prefix}/readers/{id}` | Shared (multiple readers) |

**Operations:**

```rust
// Non-blocking write lock
let lock = client.try_write_lock("my_resource").await?;

// Blocking read lock with polling (100ms intervals)
let lock = client.read_lock_with_wait("my_resource").await?;
```

## NATS Architecture

### When NATS is Used

NATS is used for:

1. **KV Cache Events**: Real-time KV cache state updates for routing
2. **Router Replica Sync**: Synchronizing router state across replicas
3. **Legacy Request Plane**: NATS-based request transport (optional)

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NATS_SERVER` | NATS server URL | `nats://localhost:4222` |

### Disabling NATS

For deployments without KV-aware routing:

```bash
# Disable NATS and KV events
python -m dynamo.frontend --no-kv-events
```

This enables "approximate mode" for KV routing without event persistence.

### Event Publishing

Components publish events to NATS subjects:

```rust
pub trait EventPublisher {
    async fn publish(&self, event: &str, data: &[u8]) -> Result<()>;
    async fn publish_serialized<T: Serialize>(&self, event: &str, data: &T) -> Result<()>;
}
```

**Subject Naming:**
```
{base_subject}.{event_name}
```

Example:
```
vllm-agg.backend.kv_cache_update
```

### Event Subscription

Components subscribe to events:

```rust
pub trait EventSubscriber {
    async fn subscribe(&self, topic: &str) -> Result<Subscriber>;
    async fn subscribe_typed<T: DeserializeOwned>(&self, topic: &str) -> Result<TypedSubscriber<T>>;
}
```

### JetStream Persistence

For durable event delivery, NATS JetStream provides:

- Message persistence
- Replay from offset
- Consumer groups for load balancing
- Acknowledgment tracking

## Key-Value Store Abstraction

Dynamo provides a unified KV store interface supporting multiple backends:

### Supported Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| `EtcdStore` | Production deployments | `ETCD_ENDPOINTS` |
| `MemoryStore` | Testing, development | Default |
| `NatsStore` | NATS-only deployments | `NATS_SERVER` |
| `FileStore` | Local persistence | File path |

### Store Interface

```rust
pub trait KvStore {
    async fn get(&self, bucket: &str, key: &str) -> Result<Option<Vec<u8>>>;
    async fn put(&self, bucket: &str, key: &str, value: &[u8]) -> Result<()>;
    async fn delete(&self, bucket: &str, key: &str) -> Result<()>;
    async fn watch(&self, bucket: &str) -> Result<WatchStream>;
}
```

### Buckets

Data is organized into logical buckets:

| Bucket | Purpose |
|--------|---------|
| `v1/instances` | Endpoint instance registry |
| `v1/mdc` | Model deployment cards |

## Typed Prefix Watcher

For type-safe watching of etcd prefixes:

```rust
// Watch and maintain HashMap of deserialized values
let watcher = watch_prefix_with_extraction::<DiscoveryInstance>(
    &etcd_client,
    "/services/vllm-agg/",
    lease_id_extractor,
    value_extractor,
).await?;

// Receive updates via watch channel
let instances = watcher.borrow();
```

**Key Extractors:**

| Extractor | Description |
|-----------|-------------|
| `lease_id()` | Use lease ID as key |
| `key_string()` | Extract key with prefix stripping |
| `full_key_string()` | Use full etcd key |

## Reliability Features

### Connection Resilience

**etcd Reconnection:**
- Exponential backoff: 50ms to 5s
- Deadline-based retry logic
- Mutex ensures single concurrent reconnect

**NATS Reconnection:**
- Built-in reconnection in NATS client
- Configurable max reconnect attempts
- Buffering during disconnection

### Lease-Based Cleanup

When a worker crashes or loses connectivity:

1. Keep-alive heartbeats stop
2. Lease expires after TTL (10 seconds)
3. All registered endpoints automatically deleted
4. Clients receive DELETE watch events
5. Traffic reroutes to healthy workers

### Transaction Safety

etcd transactions ensure atomic operations:

```rust
// Atomic create-if-not-exists
let txn = Txn::new()
    .when([Compare::create_revision(key, CompareOp::Equal, 0)])
    .and_then([Op::put(key, value, options)]);

etcd_client.txn(txn).await?;
```

This prevents race conditions in concurrent service registration.

## Operational Modes

### Kubernetes Mode (Requires Explicit Configuration)

Native Kubernetes service discovery:

```bash
# Operator explicitly sets this (not auto-detected):
export DYN_DISCOVERY_BACKEND=kubernetes

# Workers register via K8s CRDs
python -m dynamo.vllm --model Qwen/Qwen3-0.6B

# Frontend discovers workers via K8s API
python -m dynamo.frontend
```

No etcd or NATS required for basic operation when using K8s discovery.

### KV Store Mode (Global Default)

Full service discovery with etcd:

```bash
# This is the default - no configuration needed
# export DYN_DISCOVERY_BACKEND=kv_store  # (implicit)

# Workers register with etcd
python -m dynamo.vllm --model Qwen/Qwen3-0.6B

# Frontend discovers workers via etcd
python -m dynamo.frontend
```

### KV-Aware Routing (Optional)

Enable NATS for KV cache event tracking:

```bash
# Default: KV events enabled (requires NATS)
python -m dynamo.frontend --router-mode kv

# Disable KV events for prediction-based routing (no NATS)
python -m dynamo.frontend --router-mode kv --no-kv-events
```

With `--no-kv-events`:
- Router predicts cache state based on routing decisions
- TTL-based expiration and LRU pruning
- No NATS infrastructure required

## Best Practices

### 1. Use Kubernetes Discovery on K8s

The Dynamo operator automatically sets `DYN_DISCOVERY_BACKEND=kubernetes` for pods. No additional setup required when using the operator.

### 2. For Bare Metal: Deploy etcd Cluster

For bare-metal production deployments, deploy a 3-node etcd cluster for high availability.

### 3. Configure Appropriate TTLs (etcd mode)

Balance between detection speed and overhead:

- **Short TTL (5s)**: Faster failure detection, more keep-alive traffic
- **Long TTL (30s)**: Less overhead, slower detection

### 4. KV Routing Without NATS

For simpler deployments without NATS:

```bash
# Use prediction-based KV routing
python -m dynamo.frontend --router-mode kv --no-kv-events
```

This provides KV-aware routing with reduced accuracy but no NATS dependency.

## Related Documentation

- [Distributed Runtime](distributed-runtime.md) - Runtime architecture
- [Request Plane](request-plane.md) - Request transport configuration
- [Fault Tolerance](../fault-tolerance/README.md) - Failure handling
