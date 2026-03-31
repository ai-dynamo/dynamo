---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KVBM G3PB Peer Cache Plan
---

# KVBM G3PB Peer-Backed Cache Plan

**Status:** Active implementation plan

## Goal

`G3PB` (`G3PeerBaseten`) replaces the unlanded `G4` NVMe-RAID storage-agent work
with a remote peer-backed cache. The peer owns a hybrid CPU+disk cache
internally, while KVBM sees only a block-based remote cache keyed by
`sequence_hash`.

The current implementation direction is intentionally conservative:

- deterministic peer ownership via rendezvous hashing
- direct peer `offer` / `put` / `query` / `fetch`
- immutable blocks keyed by `sequence_hash`
- cache-miss fallback to recompute
- local onboarding remains `host -> device`
- transport-visible remote memory is CPU-visible staging, not remote disk

## Architecture

### Components

- **Inference worker**: computes `sequence_hash`, routes to the owning peer,
  fetches remote hits, and onboards blocks locally.
- **G3PB peer**: serves block metadata and payloads for its ownership shard.
- **Peer-local storage backend**: implements the peer cache contract. The first
  backend is in-memory; a future backend will use `foyer`.
- **Discovery / membership source**: provides the live peer set so workers can
  compute ownership locally.

### Data model

The peer cache entry is block-centric and keyed only by `sequence_hash`.
Remote identity does not include a disk index.

```rust
struct G3pbCacheEntry {
    sequence_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
    payload: Option<Vec<u8>>,
}
```

- `sequence_hash` is the canonical cache key.
- `checksum` is optional transfer-validation metadata.
- `payload` may be absent for metadata-only admission paths.

## Ownership and routing

Workers derive ownership from the live peer set:

```text
owner = rendezvous_hash(sequence_hash, active_peers)
```

This keeps routing deterministic without a global per-block index.

## Peer API

The remote API stays block-based:

```rust
trait G3pbPeerStorage {
    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<u64>;
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>);
    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<()>;
    async fn query_blocks(&self, hashes: &[u64]) -> Vec<G3pbQueryHit>;
    async fn fetch_blocks(&self, hashes: &[u64]) -> Result<Vec<G3pbTransferBlock>>;
}
```

Where:

```rust
struct G3pbPutBlock {
    sequence_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
}

struct G3pbQueryHit {
    worker_id: u64,
    sequence_hash: u64,
    size_bytes: usize,
    checksum: Option<[u8; 32]>,
}

struct G3pbTransferBlock {
    meta: G3pbPutBlock,
    payload: Vec<u8>,
}
```

## Transfer model

The intended runtime path is:

```text
local device/host <-> remote CPU-visible staging via NIXL over UCX
```

Important constraints:

- remote disk is internal to the peer cache backend
- remote disk is not a KVBM-visible transfer tier
- remote disk -> GPU direct transfer is not part of `G3PB`
- cache misses remain acceptable and degrade to recompute

## Landed implementation state

The current landed slice covers the first honest `G3PB` seam:

- `distributed/g3pb.rs` owns the peer-routing and peer-cache API
- the first peer backend is in-memory
- the standalone backend binary exposes HTTP `health`, `offer`, `put`,
  `put_payload`, `query`, and `fetch`
- the worker smoke still validates `query -> fetch -> local host register ->
  device onboard`
- `disk_block_idx` and `G4BlockIndex` are no longer part of the remote API
  identity or core block-manager state

## Next milestones

1. Keep the in-memory backend as the contract reference.
2. Add a `foyer`-backed peer-local storage implementation behind the same trait.
3. Reconnect the transport path to real NIXL/UCX `device <-> CPU` transfers.
4. Revalidate in layers:
   - host-only peer-cache smoke
   - `kvbm_nixl_transfer_smoke`
   - full worker/peer smoke with remote fetch and local onboard

## Non-goals

- turning `G3PB` into a strongly consistent storage system
- exposing peer-local disk layout through the remote API
- adding a new generic core KVBM cache tier enum before the unlanded surface
  settles
