# Standalone Indexer: Cross-Process IPC and Serialization

This document covers how the standalone indexer handles the three serialization
boundaries that arise when crossing process boundaries for `find_matches` and
event ingestion.

## The three serialization concerns

### 1. KV cache events (inbound) — ZMQ/NATS, not HTTP

Events do **not** go through the HTTP server. `subscriber.rs` subscribes to
the dynamo event plane:

```rust
let mut subscriber = EventSubscriber::for_component_with_transport(
    &worker_component,
    KV_EVENT_SUBJECT,
    transport_kind,
)
.await?
.typed::<RouterEvent>();
```

Workers publish `RouterEvent`s as pub-sub messages (ZMQ or NATS); the standalone
indexer consumes and deserializes them via `.typed::<RouterEvent>()` and calls
`indexer.apply_event(event).await` directly. Serialization is handled entirely
by the event plane transport — there is no per-event HTTP overhead and no
polling. Events flow in asynchronously as they are published.

### 2. `find_matches` requests (inbound) — two transport modes

The standalone indexer can serve queries over two transports:

**HTTP mode** (`run_server`):
- axum HTTP server at `/query` (token IDs) or `/query_by_hash` (pre-hashed block hashes)
- JSON request body; token→hash computation happens server-side
- Higher per-request overhead (~HTTP framing + JSON parse)

**Dynamo runtime mode** (`run_with_runtime`):
- Registers a dynamo `Ingress<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>>` endpoint
- Backed by velo-transports TCP/UDS with a custom 11-byte preamble framing protocol
- Zero-copy receive via `BytesMut::split_to().freeze()`; still requires struct serialization
- Lower overhead than HTTP but not zero-cost

In both modes the request payload is `Vec<LocalBlockHash>` = `Vec<u64>` — a flat
byte buffer with no heap-allocated sub-structures. This is the cheapest
possible wire representation for the input side.

### 3. `OverlapScores` (outbound) — `WireOverlapScores`

`OverlapScores` uses `FxHashMap<WorkerWithDpRank, u32>` internally. Struct keys
are not valid JSON object keys and cannot be trivially serialized. The
`WireOverlapScores` type in `indexer/types.rs` solves this by flattening to
vecs of tuples:

```rust
pub struct WireOverlapScores {
    pub scores:      Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
    pub tree_sizes:  Vec<(WorkerWithDpRank, usize)>,
}
```

The `From<OverlapScores>` conversion is `O(n)` in the number of workers — in
practice worker counts are in the tens to hundreds, not thousands, so this is
negligible. On the receiver side, `From<WireOverlapScores>` reconstructs the
`FxHashMap` by iterating the vec.

## End-to-end latency budget

The dominant cost in a cross-process `find_matches` call is the network
round-trip, not serialization:

| Leg | Approximate cost |
|-----|-----------------|
| `Vec<LocalBlockHash>` serialize | ~1–5 µs (flat u64 array) |
| Loopback UDS round-trip | ~50 µs |
| Localhost TCP round-trip | ~100–200 µs |
| CRTC traversal (in-process baseline) | ~50–200 µs |
| `WireOverlapScores` flatten + deserialize | ~5–20 µs |

For comparison, in-process `spawn_blocking` scatter overhead under load is
~200–400 µs (futex wakeup latency). At low load, in-process is faster; at high
concurrency with blocked threads, the cross-process UDS round-trip can be
competitive.

## Implications for a multi-process sharded design

The standalone indexer provides all the pieces needed to build a multi-process
sharded indexer:

- **Event routing**: events are already broadcast to all subscribers on the
  event plane; shard processes selectively apply only events for their assigned
  workers.
- **Query transport**: `IndexerQueryRequest` / `IndexerQueryResponse` are the
  ready-made wire types.
- **Score merging**: `merge_scores` in `sharded_concurrent.rs` already
  implements the union-merge logic needed after gathering responses from shards.
- **Worker assignment**: the coordinator needs to maintain a
  `WorkerId → shard_process` table and forward events accordingly (or let each
  shard self-filter from the broadcast stream).
