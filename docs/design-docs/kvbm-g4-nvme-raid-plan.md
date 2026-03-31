---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM G4 NVMe RAID Plan
---

# KVBM G4 NVMe RAID-Backed Storage Agent Plan

**Status:** Draft plan

This document proposes a first `G4` implementation for KVBM using NVMe RAID-backed storage agents. The design intentionally favors a simple cache architecture over a fully distributed storage system:

- deterministic ownership per block
- selective redundancy for hot prefix blocks
- no background repair
- worker-to-agent direct query and fetch
- event plane used only as optional fallback or metadata hinting, not as the primary lookup path
- pinned-host staging on the storage agent for remote transfers
- NIXL over UCX as the recommended remote transfer mechanism

## Overview

The current KVBM implementation actively supports `G1/G2/G3` flows. `G4` exists in the type system and design docs, but not yet as a completed runtime path. This plan defines a practical first `G4` path that works on local high-throughput storage nodes without requiring a shared filesystem or a distributed metadata mesh.

The key idea is:

1. Workers derive `sequence_hash` locally from token blocks.
2. Workers compute the owning `G4` storage agent from a deterministic hash over the active agent set.
3. Workers query the owner directly for exact block existence.
4. On hit, workers fetch the block payload from that owner and onboard it locally.
5. On miss or transfer failure, workers treat the block as a cache miss and recompute.
6. The first `N` blocks of a sequence may be replicated to two owners to absorb disproportionate read and write load on very hot prefix blocks.

## Design Goals

1. **Simple correctness model** - `G4` is a cache, not a source-of-truth store. Misses are acceptable and fall back to recompute.
2. **High-throughput local storage** - Each storage node uses a local NVMe RAID volume for strong read bandwidth and simpler capacity management.
3. **Deterministic routing** - Workers can compute block ownership locally without a global per-block metadata service.
4. **Low coordination overhead** - No cross-agent lookup gossip or background replica repair in the hot path.
5. **Incremental integration** - The first version should fit the existing KVBM architecture without requiring a full rework of `OffloadManager`.
6. **Hot-prefix load spreading** - Very early blocks should be able to use limited redundancy so reads and puts for common prefixes do not overload a single agent.

## Non-Goals

- No background repair
- No prefix-tree or radix-tree lookup as a primary requirement
- No event-plane dependence for correctness
- No shared POSIX mount as the main data path
- No attempt to make `G4` a strongly consistent distributed database

## Selective Redundancy Policy

The first version should support limited redundancy for the first `N` blocks of a sequence.

Recommended starting policy:

- `N = 64`
- redundancy factor `2` for positions `0..63`
- redundancy factor `1` for all later positions

Rationale:

- the earliest blocks in a sequence are the most likely to be reused across requests
- those same blocks can attract disproportionate read load and repeated `put()` traffic
- selective redundancy limits read and write amplification to the high-value portion of the prefix while keeping storage and coordination costs bounded

This does **not** imply a repair mechanism. If one replica disappears, the system may continue with the surviving copy or refill the block later from recompute or a future `put()`.

## Architecture

### Components

- **Inference Worker**: Computes block hashes locally, queries/fetches from the owner, and onboards blocks into local KVBM tiers.
- **G4 Storage Agent**: Owns a shard of blocks, stores payloads on local NVMe RAID storage, maintains local metadata, and serves query/fetch/put APIs.
- **Discovery Plane**: Publishes the live set of storage agents and their endpoints.
- **Optional Event Plane Subscriber**: Consumes block lifecycle events to warm metadata or support fallback observability workflows.

### Data Model

Each block is treated as an immutable object keyed by `sequence_hash`.

Suggested local metadata record:

```rust
struct G4BlockMeta {
    sequence_hash: u64,
    block_hash: u64,
    position: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
    disk_path: String,
    last_access_unix_ms: u64,
    lease_until_unix_ms: Option<u64>,
    block_size_tokens: usize,
    model_signature: String,
}
```

Only exact-block lookup is required for the first version. A local `HashMap<SequenceHash, G4BlockMeta>` or embedded KV store is sufficient.

## Hashing and Block Identity

KVBM already has two relevant hash concepts:

- **`BlockHash`**: content-only hash of the tokens in a single block
- **`SequenceHash`**: parent-aware chained hash used as the actual reusable block identity

Current implementation details:

- `compute_hash_v2()` uses `xxh3_64_with_seed(...)`
- `block_hash = hash(tokens, salt_hash)`
- first block: `sequence_hash = block_hash`
- subsequent blocks: `sequence_hash = hash([parent_sequence_hash, block_hash], salt_hash)`

For `G4`, the primary lookup key should be `sequence_hash`.

`block_hash` may still be stored in metadata for debugging, validation, or future secondary indexing, but it should not replace `sequence_hash` as the ownership and lookup key.

Checksums are separate from identity. They should be treated as transfer-validation metadata, not as the canonical KVBM block key.

### Optional Content Hash Validation

`xxh3`-based block and sequence hashes are already fast enough to serve as the main identity path. The `G4` plan should also allow optional content-hash validation on transfers.

Recommended policy:

- identity key: `sequence_hash`
- auxiliary content field: `block_hash`
- optional transfer validation field: stronger content checksum or content hash, validated on both sides when enabled

This allows the system to stay fast by default while preserving the option to enable stricter validation for debugging, canary deployments, or corruption-sensitive environments.

## Routing and Ownership

### Ownership Rule

Each block has either one or two owners depending on sequence position:

```text
owners = top_k_rendezvous_hash(sequence_hash, active_storage_agents, k)
```

Where:

- `k = 2` for the first `N` blocks of a sequence
- `k = 1` for all later blocks

Rendezvous hashing is preferred because it is simple to compute locally, naturally supports top-`k` owner selection, and handles membership changes cleanly.

For normal reads, the worker may try either owner for the redundant prefix blocks. For writes, the worker may write to both owners for the redundant prefix region and to a single owner for the non-redundant suffix.

### Discovery Dependency

Workers do not need a global block index, but they do need:

- the current live storage-agent membership set
- endpoint addresses for those agents
- a ring or membership epoch to reason about topology changes

Discovery is therefore responsible for agent liveness and endpoint publication, not for per-block lookup.

## Storage Layout on NVMe RAID

Each storage agent writes blocks to its own local NVMe RAID volume. The simplest initial layout is:

- hash-based subdirectories to avoid oversized single directories
- immutable block payload files
- separate metadata store for lookup

Example:

```text
/nvme-raid/kvbm-g4/ab/cd/<sequence_hash>.blk
/nvme-raid/kvbm-g4/ef/01/<sequence_hash>.blk
```

The agent should treat the RAID volume as a single fast local store rather than managing per-disk block placement itself.

## APIs

The first version should expose block-centric APIs, not request-centric ones.

Suggested interface:

```rust
trait G4StorageAgent {
    async fn query_blocks(&self, hashes: Vec<u64>) -> Result<Vec<G4BlockHit>>;
    async fn fetch_blocks(&self, hashes: Vec<u64>) -> Result<Vec<G4BlockPayload>>;
    async fn put_blocks(&self, blocks: Vec<G4PutBlock>) -> Result<()>;
}
```

Suggested types:

```rust
struct G4BlockHit {
    sequence_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
}

struct G4PutBlock {
    sequence_hash: u64,
    block_hash: u64,
    position: u64,
    size_bytes: usize,
    bytes: bytes::Bytes,
    checksum: Option<[u8; 32]>,
}
```

The API intentionally does not include `has_request()`. Requests are transient scheduler concepts; reusable cache state is block-based.

If payload sizing or flow control becomes a problem, `put()` may be split into a metadata-first and bytes-second flow:

```rust
trait G4StorageAgent {
    async fn put(&self, block: G4PutBlockMeta) -> Result<PutTicket>;
    async fn put_bytes(&self, ticket: PutTicket, bytes: bytes::Bytes) -> Result<()>;
}
```

This keeps the plan flexible without forcing a two-phase write path from day one.

## Transfer Plan

### Recommended Remote Transfer Path

For the first version, remote block transfer should use:

- **Control plane:** direct RPC-style query and fetch requests to the owning storage agent
- **Data plane:** NIXL over UCX
- **Staging model:** storage agent reads from NVMe RAID into pinned host memory, then transfers to the worker
- **Initiation model:** transfer may be initiated from either side depending on what fits KVBM integration best

The recommended read path is:

```text
NVMe RAID file -> pinned host buffer on storage agent -> NIXL/UCX transfer -> worker host/device staging -> local onboard
```

This is intentionally conservative.

The first version should not depend on direct remote-disk-to-device transfer. Pinned-host staging is the simpler and safer starting point.

Transfer initiation should remain flexible:

- **pull-style**: worker requests bytes and the storage agent sends them
- **push-style**: worker provides transfer descriptors or targets and the storage agent pushes into them

KVBM already has abstractions where either side can conceptually drive transfer setup. The `G4` integration should preserve that flexibility instead of hard-coding a single initiator model.

### Why This Transfer Path

- It aligns with the current transfer direction already favored in Dynamo and TensorRT-LLM runtime docs.
- It avoids coupling the first `G4` implementation to remote GDS assumptions.
- It keeps the storage agent responsible for reading local RAID-backed payloads.
- It allows workers to keep using familiar host-to-device onboard paths locally.

## Data Flow

### Write Path

When a worker produces a registered block that should be materialized in `G4`:

1. Worker computes `sequence_hash`.
2. Worker computes one or two owning storage agents from discovery membership, depending on position.
3. Worker sends `put_blocks()` to the selected owner set.
4. Owners persist the payload to local NVMe RAID storage and record metadata locally.
5. On success, the block is available in `G4`.
6. On failure, the block is simply not cached remotely on the failed target.

This write path is best-effort. Since `G4` is a cache, write failure does not affect inference correctness.

### Read Path

When a worker wants to reuse blocks from `G4`:

1. Worker derives candidate `sequence_hash` values locally from token blocks.
2. Worker computes one or two owners for each hash, depending on position.
3. Worker sends `query_blocks()` to one owner, or to either owner in the redundant prefix region.
4. For hits, worker sends `fetch_blocks()`.
5. The storage agent reads the block from NVMe RAID into pinned host memory.
6. The storage agent transfers the payload using NIXL/UCX.
7. Worker writes fetched payloads into host or device staging and onboards them into local KVBM pools.
8. On miss or transfer failure, the worker recomputes the block locally.

### Event Plane Usage

The event plane is not required for correctness in this design.

It may still be used for:

- metrics
- observability
- warming a local metadata cache
- offline reconciliation or debugging

But the primary lookup path is direct worker-to-owner query.

## Failure Model

### Storage-Agent Unavailability

If the owning storage agent is unavailable:

- `query_blocks()` fails or times out
- for redundant prefix blocks, the worker may query the second owner
- worker treats the block as a cache miss
- worker recomputes locally

No background repair is required. If a redundant prefix block is missing from one owner, the system may continue using the other owner or refill it on a future `put()`.

### Transfer Failure

If `fetch_blocks()` fails in transit:

- worker discards the failed transfer
- worker treats the block as a cache miss
- worker recomputes locally

The system should not mark the block as successfully fetched unless the full payload length and checksum are validated.

### Write Failure

If `put_blocks()` fails:

- the block remains absent from `G4`
- for redundant prefix blocks, one owner may succeed while the other fails
- no repair is attempted
- later reads use whatever copy exists, or fall back to recompute

## Metadata Store Choice

The first version does not require a distributed metadata database.

Recommended choices:

- in-memory `HashMap` for initial bring-up
- local embedded KV store later if restart persistence is needed

Examples of acceptable local metadata stores:

- `HashMap`
- `hashbrown`
- `sled`
- `SQLite`
- `RocksDB`
- `LMDB`

The metadata store is local to each storage agent because ownership is deterministic.

## Why Not Use a Distributed DB as the Payload Store

The first version should avoid storing actual KV payload bytes in a database because that typically introduces:

- write amplification
- compaction pressure
- worse large-blob handling
- unnecessary transactional overhead

Payload bytes should live on local NVMe-backed files as immutable blobs. The metadata database should only index them.

## Integration Plan

### Phase 1: Design and Interface

- Define the `G4` storage-agent API
- Define worker-to-agent message types
- Define hashing and ownership rules
- Define local disk layout and metadata schema
- Define the pinned-host transfer path and checksum validation rules
- Define the selective redundancy policy for the first `N` blocks
- Decide whether the first implementation uses `put_blocks()` only or supports `put()` + `put_bytes()`

### Phase 2: Storage Agent Bring-Up

- Register storage-agent endpoints in discovery
- Add local metadata index
- Add `query_blocks()`, `fetch_blocks()`, and `put_blocks()`
- Persist block payloads to NVMe RAID-backed local storage
- Add pinned-host buffer management for fetch responses
- Add support for dual-owner writes for the first `N` blocks

### Phase 3: Worker Integration

- Derive `sequence_hash` locally in the worker-side path
- Add owner routing based on discovery membership
- Add direct query/fetch before local recompute
- Onboard fetched blocks into local KVBM tiers
- Add alternate-owner fallback for the redundant prefix region

### Phase 4: Policy and Observability

- Decide when blocks should be written to `G4`
- Add metrics for query hit rate, fetch latency, put latency, and transfer failures
- Optionally subscribe to event-plane updates as a secondary metadata source
- Add metrics split by early-prefix redundant region versus non-redundant suffix

## Open Questions

- Should `put_blocks()` happen immediately on block registration, or only after an offload threshold is reached?
- Should the first `N` blocks be replicated synchronously to both owners, or best-effort to the second owner?
- Should content-hash validation be off by default and enabled only for selected clusters or debug modes?
- Should workers always fetch into host pinned memory first, or should the interface allow later direct device-target transfer?
- How should membership churn be handled during long fetches: strict epoch check, or best-effort with retry?
- Should `G4` materialization be tied to KVBM offload policy, or managed by a separate backend policy layer?

## Future Work

- Add prefix-aware lookup if exact-block probing proves too expensive
- Add local block compaction and disk-space-aware eviction
- Add direct device-target remote transfer when transport and capability checks are mature enough
- Add configurable redundancy windows beyond the initial first-`N` policy

## References

- [KVBM Design](./kvbm-design.md)
- [Discovery Plane](./discovery-plane.md)
- [Distributed Runtime](./distributed-runtime.md)
- [TensorRT-LLM KV Cache Transfer](../backends/trtllm/trtllm-kv-cache-transfer.md)
