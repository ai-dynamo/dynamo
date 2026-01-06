# Distributed Registry

A pluggable, high-performance registry for tracking and deduplicating data blocks across distributed workers.

## Overview

When running distributed workloads where multiple workers may store data to shared storage (S3, GCS, local filesystem, etc.), coordination is needed to avoid redundant storage. Without it, multiple workers might store the same data blocks, wasting storage and bandwidth.

The **Distributed Registry** solves this by providing a centralized catalog that tracks which blocks have been stored, enabling **cross-worker deduplication**.

```
+-------------+     +-------------+     +-------------+
|  Worker 1   |     |  Worker 2   |     |  Worker 3   |
+------+------+     +------+------+     +------+------+
       |                   |                   |
       +-------------------+-------------------+
                           |
                    +------v------+
                    |   Registry  |
                    |     Hub     |
                    +------+------+
                           |
                    +------v------+
                    |   Storage   |
                    |  (any type) |
                    +-------------+
```

## Key Concepts

### Deduplication Flow

1. **Worker wants to store blocks** - Asks registry "can I store these hashes?"
2. **Registry checks** - Returns `Granted`, `AlreadyStored`, or `Leased` for each hash
3. **Worker stores only `Granted` blocks** - Skips duplicates
4. **Worker registers stored blocks** - Registry records them for future queries

### Lease-Based Claiming

To prevent race conditions where two workers try to store the same block simultaneously, the hub uses **lease-based claiming**:

- `Granted` - You have exclusive rights to store this block (lease granted)
- `AlreadyStored` - Another worker already stored it (skip)
- `Leased` - Another worker is currently storing it (wait or skip)

Leases expire after a configurable timeout (default: 30s) if the worker fails to complete the upload.

```rust
use registry::core::{hub, HashMapStorage, NoMetadata};
use std::time::Duration;

// Configure lease TTL using the builder
let storage = HashMapStorage::<u64, u64>::new();
let registry_hub = hub::<u64, u64, NoMetadata, _>(storage)
    .lease_ttl(Duration::from_secs(60))
    .lease_cleanup_interval(Duration::from_secs(10))
    .build();

// Access lease stats
let stats = registry_hub.stats();
println!("Active leases: {}", stats.lease_stats.active);
println!("Leases granted: {}", stats.lease_stats.granted);
println!("Leases expired: {}", stats.lease_stats.expired);
```

## Architecture

The registry uses a **pluggable architecture** with trait-based abstractions:

```
+-----------------------------------------------------------+
|                        Client                              |
|  +---------+  +---------+  +---------+  +------------+    |
|  |   Key   |  |  Value  |  |Metadata |  |   Codec    |    |
|  |  (u64)  |  |  (u64)  |  |  (None) |  |  (Binary)  |    |
|  +----+----+  +----+----+  +----+----+  +-----+------+    |
|       +--------------+-----------+-----------+            |
|                      |                                    |
|               +------v------+                             |
|               |  Transport  |                             |
|               |  (ZMQ/etc)  |                             |
|               +-------------+                             |
+-----------------------------------------------------------+
                       |
                +------v------+
                |     Hub     |
                |  (Server)   |
                | +---------+ |
                | | Storage | |
                | |(HashMap)| |
                | +---------+ |
                +-------------+
```

### Core Traits

| Trait | Purpose | Implementations |
|-------|---------|-----------------|
| `RegistryKey` | Hash/identifier for blocks | `u64`, `Key128`, `PositionalKey`, `CompositeKey` |
| `RegistryValue` | What's stored with the key | `u64`, `StorageLocation` |
| `RegistryMetadata` | Optional metadata | `NoMetadata`, `TimestampMetadata`, `PositionMetadata` |
| `Storage` | Key-value backend | `HashMapStorage` |
| `Eviction` | Cache eviction policy | `NoEviction`, `TailEviction` |
| `RegistryTransport` | Network communication | `InProcessTransport`, `ZmqTransport` |
| `RegistryCodec` | Message serialization | `BinaryCodec` (versioned) |

### ZMQ Communication Pattern

```
Client                              Hub
  |                                  |
  |<------- DEALER/ROUTER --------->|  (Queries: request/response)
  |                                  |
  |<------- PUSH/PULL ------------>|  (Registrations: fire-and-forget)
  |                                  |
```

- **DEALER/ROUTER**: Bidirectional request/response for queries
- **PUSH/PULL**: Many-to-one for fire-and-forget registrations

## Usage

### Using the Builder Pattern

The recommended way to create clients and hubs:

```rust
use registry::core::{client, hub, HashMapStorage, NoMetadata, ZmqTransport};
use std::time::Duration;

// Create a client with any transport
let transport = ZmqTransport::connect_to("hub-host", 5555, 5556)?;
let registry_client = client::<u64, u64, NoMetadata, _>(transport)
    .batch_size(50)
    .batch_timeout(Duration::from_millis(20))
    .build();

// Create a hub with any storage backend
let storage = HashMapStorage::<u64, u64>::new();
let registry_hub = hub::<u64, u64, NoMetadata, _>(storage)
    .lease_ttl(Duration::from_secs(60))
    .lease_cleanup_interval(Duration::from_secs(10))
    .build();
```

### Client Operations

```rust
use registry::core::{client, NoMetadata, Registry};

let registry_client = client::<u64, u64, NoMetadata, _>(transport).build();

// Check what can be stored
let result = registry_client.can_offload(&[hash1, hash2, hash3]).await?;
// result.can_offload: hashes you should store (lease granted)
// result.already_stored: skip these
// result.leased: another worker is storing these (wait or skip)

// After storing, register the blocks
registry_client.register(&[(hash1, value1, NoMetadata)]).await?;
registry_client.flush().await?;

// Find matching blocks
let matches = registry_client.match_prefix(&hashes).await?;
```

### Serving a Hub

```rust
use registry::core::{hub, HashMapStorage, NoMetadata, ZmqHubTransport, ZmqHubConfig};

let storage = HashMapStorage::<u64, u64>::new();
let registry_hub = hub::<u64, u64, NoMetadata, _>(storage)
    .lease_ttl(Duration::from_secs(30))
    .build();

// Serve with ZMQ transport
let mut transport = ZmqHubTransport::bind(ZmqHubConfig::default())?;
registry_hub.serve(&mut transport).await?;
```

### Batch Registration with Timeout

The client batches registrations for efficiency. Batches are flushed when:
- Batch size threshold is reached (default: 100)
- Batch timeout expires (default: 10ms)

```rust
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use registry::core::{client, NoMetadata};

let registry_client = Arc::new(
    client::<u64, u64, NoMetadata, _>(transport)
        .batch_size(50)
        .batch_timeout(Duration::from_millis(20))
        .build()
);

// Start background flush task
let cancel = CancellationToken::new();
let _flush_task = registry_client.start_batch_flush_task(cancel.clone());

// Registrations are automatically batched and flushed
registry_client.register(&[(hash1, value1, NoMetadata)]).await?;

// On shutdown
cancel.cancel();
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_REGISTRY_ENABLE` | Enable distributed registry | `false` |
| `DYN_REGISTRY_CLIENT_QUERY_ADDR` | Hub query address | `tcp://localhost:5555` |
| `DYN_REGISTRY_CLIENT_REGISTER_ADDR` | Hub registration address | `tcp://localhost:5556` |
| `DYN_REGISTRY_CLIENT_NAMESPACE` | Client namespace identifier | `default` |
| `DYN_REGISTRY_CLIENT_BATCH_SIZE` | Batch size before flush | `100` |
| `DYN_REGISTRY_CLIENT_BATCH_TIMEOUT_MS` | Batch timeout in ms | `10` |
| `DYN_REGISTRY_CLIENT_REQUEST_TIMEOUT_MS` | Request timeout in ms | `5000` |
| `DYN_REGISTRY_CLIENT_LOCAL_CACHE` | Local cache capacity | `0` |
| `DYN_REGISTRY_HUB_QUERY_ADDR` | Hub bind address for queries | `tcp://*:5555` |
| `DYN_REGISTRY_HUB_REGISTER_ADDR` | Hub bind address for registrations | `tcp://*:5556` |
| `DYN_REGISTRY_HUB_CAPACITY` | Max entries in registry | `1000000` |
| `DYN_REGISTRY_HUB_LEASE_TIMEOUT_SECS` | Lease TTL in seconds | `30` |

### Backpressure Configuration

ZMQ sockets use high-water marks (HWM) to prevent unbounded memory growth:

```rust
use registry::core::ZmqTransportConfig;

// Default: 10,000 messages per socket
let config = ZmqTransportConfig::default();

// Custom HWM for memory-constrained systems
let config = ZmqTransportConfig::new(query_addr, push_addr)
    .with_hwm(1000);  // Buffer only 1,000 messages

// Fine-grained control
let config = ZmqTransportConfig::new(query_addr, push_addr)
    .with_dealer_hwm(5000)   // Queries
    .with_push_hwm(20000);   // Registrations
```

| HWM Value | Behavior Under Load |
|-----------|---------------------|
| Low (1K) | May drop messages under heavy load |
| Default (10K) | Balanced for most workloads |
| High (100K) | Higher memory usage, higher latency |

## Protocol Versioning

The wire protocol includes a version byte:

```
[version:1][type:1][count:4][...data...]
```

Current version: `PROTOCOL_VERSION = 1`

Messages without a valid version byte are rejected.

## Error Handling

The library provides structured error types:

```rust
use registry::core::{RegistryError, RegistryResult};

match result {
    Err(RegistryError::Timeout { duration_ms }) => {
        println!("Request timed out after {}ms", duration_ms);
    }
    Err(RegistryError::LeaseConflict { key_debug }) => {
        println!("Lease conflict for key: {}", key_debug);
    }
    Err(RegistryError::DecodeError { context, expected, got }) => {
        println!("Decode error in {}: expected {}, got {}", context, expected, got);
    }
    _ => {}
}
```

## Eviction Strategies

### TailEviction (Prefix-Aware)

For sequence-based data where blocks form chains:

```
Block 1 -> Block 2 -> Block 3 -> Block 4
(root)                          (leaf)
```

When evicting, remove **leaves first** (tail-first) to avoid orphaning children:

```rust
use registry::core::{HashMapStorage, TailEviction};

let storage = HashMapStorage::new();
let evictable = TailEviction::new(storage, 10_000); // capacity

// Insert with parent tracking
evictable.insert_with_parent(block_hash, value, Some(parent_hash));

// Eviction removes deepest leaves first
let evicted = evictable.evict(100);
```

## Testing

```bash
# Run all registry tests
cargo test --lib -- registry::core

# Run ZMQ integration tests (requires network)
cargo test --lib -- test_zmq --ignored
```

## Module Structure

```
registry/
+-- mod.rs              # Module exports
+-- config.rs           # Configuration structs
+-- README.md           # This file
+-- core/
    +-- mod.rs          # Core exports
    +-- key.rs          # RegistryKey trait + implementations
    +-- value.rs        # RegistryValue trait + implementations
    +-- metadata.rs     # RegistryMetadata trait + implementations
    +-- storage.rs      # Storage trait + HashMapStorage
    +-- eviction.rs     # Eviction trait + TailEviction
    +-- transport.rs    # RegistryTransport trait + InProcessTransport
    +-- codec.rs        # RegistryCodec trait + BinaryCodec (versioned)
    +-- registry.rs     # Registry trait + RegistryClient
    +-- hub.rs          # Generic RegistryHub with lease management
    +-- lease.rs        # LeaseManager for race condition prevention and unecessary writes
    +-- error.rs        # RegistryError types
    +-- zmq_transport.rs  # ZMQ client transport (with HWM)
    +-- zmq_hub.rs        # ZMQ hub server
    +-- hub_transport.rs  # Hub transport traits (with HWM)
    +-- builder.rs        # Builder patterns
    +-- tests.rs          # Integration tests
```

## License

Apache-2.0
