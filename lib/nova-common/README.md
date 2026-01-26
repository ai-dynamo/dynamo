# dynamo-nova-common

Common types for the Nova distributed systems stack.

## Overview

This crate provides the foundational types used across Nova for identity and addressing. The design prioritizes:

- **Compact representations** for embedding in fixed-size handles
- **Transport-agnostic addressing** without enumerating all possible transports
- **KV-store friendly** serialization using opaque `Bytes`

## Identity Types

### InstanceId

`InstanceId` is a UUID-based identifier that serves as the **source of truth** for identifying a running Nova instance. It is used for:

- Transport-level routing
- Discovery registration
- Peer management

```rust
use dynamo_nova_common::InstanceId;

let instance_id = InstanceId::new_v4();
let uuid = instance_id.as_uuid();
```

### WorkerId

`WorkerId` is a deterministic 64-bit identifier derived from `InstanceId` via xxh3 hash. The compact representation enables embedding worker identity into fixed-size handles.

**Design rationale**: A `u128` handle can encode:
- 64 bits for `WorkerId`
- 64 bits for additional data (sequence numbers, flags, etc.)

This value-semantics approach simplifies passing identity through systems that work with fixed-size integers.

```rust
use dynamo_nova_common::InstanceId;

let instance_id = InstanceId::new_v4();
let worker_id = instance_id.worker_id();  // Deterministic derivation

// Embed in a u128 handle
let other_data: u128 = 42;
let handle: u128 = (worker_id.as_u64() as u128) << 64 | other_data;
```

The derivation is always consistent—calling `worker_id()` multiple times returns the same value.

## Address Types

### WorkerAddress

`WorkerAddress` is an opaque byte container holding transport endpoint information. Internally, it's a MessagePack-encoded map of `TransportKey -> Bytes`, but this structure is intentionally hidden from consumers.

**Key design decisions**:

1. **Opaque values**: Transport endpoints are stored as raw bytes. They could be simple strings (`"tcp://127.0.0.1:5555"`) or complex serialized objects. The interpretation is left to the transport implementation.

2. **No transport enum**: Rather than defining an enum of all possible transports with their configurations, we use string keys (`"tcp"`, `"rdma"`, `"grpc"`, etc.). This allows transports to be added without modifying the common types.

3. **KV-store friendly**: The entire address serializes to a `Bytes` blob, suitable for storage in etcd, Redis, or any key-value store without schema changes.

```rust
use dynamo_nova_common::WorkerAddress;
use std::collections::HashMap;

// Addresses are typically constructed by nova-backend transport builders,
// but here we demonstrate the consumer API:
let mut map = HashMap::new();
map.insert("tcp".to_string(), b"tcp://127.0.0.1:5555".to_vec());
map.insert("rdma".to_string(), b"rdma://10.0.0.1:6666".to_vec());
let encoded = rmp_serde::to_vec(&map).unwrap();
let address = WorkerAddress::from_encoded(encoded);

// Reading an address (consumer perspective)
let transports = address.available_transports().unwrap();  // ["tcp", "rdma"]
let tcp_endpoint = address.get_entry("tcp").unwrap();      // Some(Bytes)
```

### TransportKey

A type-safe wrapper around transport identifiers. Provides zero-cost abstraction over `Arc<str>` with efficient cloning and HashMap compatibility.

```rust
use dynamo_nova_common::TransportKey;
use std::collections::HashMap;

let key = TransportKey::from("tcp");
let key2: TransportKey = "rdma".into();

// Works with HashMap lookups via Borrow<str>
let mut map = HashMap::new();
map.insert(TransportKey::from("tcp"), "127.0.0.1:5555");
assert!(map.get("tcp").is_some());  // &str lookup works
```

### PeerInfo

Combines `InstanceId` and `WorkerAddress` into a single structure representing a discoverable peer. This is the primary type exchanged during peer discovery and registration.

```rust
use dynamo_nova_common::{InstanceId, PeerInfo, WorkerAddress};
use std::collections::HashMap;

let instance_id = InstanceId::new_v4();
let map: HashMap<String, Vec<u8>> = HashMap::new();
let encoded = rmp_serde::to_vec(&map).unwrap();
let worker_address = WorkerAddress::from_encoded(encoded);

let peer_info = PeerInfo::new(instance_id, worker_address);
assert_eq!(peer_info.instance_id(), instance_id);
```

## Address Construction

`WorkerAddress` instances are constructed by transport builders in `nova-backend`. Each transport (TCP, gRPC, NATS, UCX, etc.) contributes its endpoint data:

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  TCP Transport  │     │ gRPC Transport  │     │  UCX Transport  │
│  Builder        │     │  Builder        │     │  Builder        │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │ "tcp" -> endpoint     │ "grpc" -> endpoint    │ "ucx" -> blob
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    WorkerAddress       │
                    │  (MessagePack map)     │
                    │                        │
                    │  tcp  -> bytes         │
                    │  grpc -> bytes         │
                    │  ucx  -> bytes         │
                    └────────────────────────┘
```

When a Nova client receives a `PeerInfo`, it can:
1. Check `available_transports()` to see what's supported
2. Extract the relevant endpoint via `get_entry(key)`
3. Register the peer with its own transports

This design decouples the common types from specific transport implementations.

## Wire Format

- **InstanceId**: Serializes as a UUID string (JSON) or 16 bytes (binary)
- **WorkerId**: Serializes as a u64
- **WorkerAddress**: Serializes as a byte array (the MessagePack-encoded map)
- **PeerInfo**: Serializes as a struct with `instance_id` and `worker_address` fields

All types implement `serde::Serialize` and `serde::Deserialize`.
