<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cuckoo KV Indexer

This directory contains the Cuckoo-filter (CKF) producer, publication protocol, and
read-optimized global consumer used by the DC KV Relay architecture. The implementation has two
different storage roles over one compatible packed-bucket format:

- `DcCkfState` is the single-owner, mutable producer for one DC-local pool.
- `GlobalCkfIndexer` is the concurrent, transposed consumer for one indexer domain across as many
  as 16 DC pool lanes.

The producer and consumer deliberately have different ownership models. The producer needs exact
full-hash ownership to interpret removals. The consumer needs compact, concurrent prefix search
across DCs and stores only the lossy CKF projection.

For the higher-level Relay and recovery model, see
[Multi-DC KV Routing and the DC Relay](https://github.com/ai-dynamo/dynamo/blob/main/docs/components/router/multi-dc-kv-routing.md).

## Core concepts

An `IndexerDomainId` identifies one logical KV indexer namespace. A `PoolId` adds a stable `DcId`
and identifies exactly one DC-local producer and publication stream:

```text
PoolId = (IndexerDomainId, DcId)
```

Native runtime endpoints are bindings for a pool, not part of the CKF protocol identity. A global
consumer owns one immutable indexer domain. Each configured lane belongs to a distinct pool and
therefore a distinct DC within that domain.

## Producer invariants

`DcCkfState` keeps the exact state needed to make mutation decisions:

- `member_blocks[(worker, dp_rank)]` records exact full-hash ownership.
- `dc_refcounts[full_hash]` equals the number of members that own the hash.
- A full hash contributes one physical fingerprint to the CKF while its refcount is nonzero.
- Additional owners change the refcount without inserting another CKF copy.
- Different full hashes that map to the same representation contribute separate physical copies.
- An unknown `(member, full_hash)` removal is an idempotent no-op. It never deletes by fingerprint.

The CKF is a lossy projection, not the authority. A packed fingerprint has no full-hash identity,
so exact ownership and refcounts must remain on the producer side.

Store and remove transitions are consequently:

| Exact transition | CKF operation |
| --- | --- |
| First owner, `0 -> 1` | Insert one fingerprint |
| Additional owner, `n -> n+1` | None |
| Remaining owner, `n -> n-1` | None |
| Final owner, `1 -> 0` | Remove one fingerprint |

The actor processes blocks in event order. A failed block is unchanged, successful sibling blocks
remain applied, later blocks are still attempted, and the first error is retained for
observability.

## Producer storage and relocation

Each bucket is one packed `u64` containing four 16-bit fingerprints. A full hash deterministically
produces one nonzero fingerprint and an unordered pair of candidate buckets. Physical placement
within those buckets is not deterministic.

The retained producer is actor-owned and therefore mutates ordinary packed cells without internal
locking. Cuckoo insertion may relocate resident fingerprints with bounded forward kicks and rolls
those writes back if no path succeeds. `max_kicks` is a bounded-search limit; reaching it does not
prove that the table is full.

Capacity exhaustion is an observable, pre-commit omission for the affected block. It does not add
an exact ownership edge, change a refcount, dirty a bucket, fence the pool, or retire the consumer
lane. Service continues, a later remove is an unknown no-op, and a later store may retry after
occupancy changes. This permits stable false negatives relative to the source cache without
corrupting tracked producer state.

## Publication protocol

The producer tracks bucket indices dirtied by successful physical mutations. Dirty tracking is a
notification mechanism: when publication runs, the actor samples the current absolute `u64` value
for each dirty bucket. Net reversions can therefore disappear from a batch.

`DcCkfPublisher` is the serialized stream owner for one pool. It adds the current producer
identity, lane lease, and sequence envelope:

```text
DcCkfSnapshot { identity, lease, sequence, buckets }

DcCkfDelta {
    identity,
    lease,
    base_sequence,
    sequence,
    absolute_bucket_images,
}
```

The actor samples an ordinary batch only after a complete command, so one batch cannot split an
event, rank clear, or relocation. Publication cadence may coalesce several complete commands.
Sequence numbers count emitted batches, not events, and exist only for FIFO continuation and gap
detection.

A barrier snapshot drains pending publication before copying the complete producer table. If it is
tagged with terminal sequence `N`, the first continuation delta must be `N -> N+1`. An exact drain
marker also carries `N`, which detects a lost final delta even when no later delta exposes a gap.

## Global transposed consumer

`GlobalCkfIndexer` stores buckets in bucket-major, lane-minor order:

```text
bucket 0: [dc lane 0, dc lane 1, ...]
bucket 1: [dc lane 0, dc lane 1, ...]
...
```

This transposition lets one probe compare the same candidate buckets across all ready DC lanes.
Queries return the matching `PoolId` and prefix depth for each lane; runtime code resolves the pool
to a serving endpoint afterward.

Each lane has one logically serialized ingestor. The Flume ingestion pool assigns a lane to a
stable worker, so snapshots, deltas, and drain markers for that lane cannot interleave. Different
lanes and query threads can proceed concurrently.

Only the packed bucket words and ready-lane mask are shared with queries:

- Each consumer bucket image is installed with one atomic `u64` store.
- Snapshot activation publishes readiness with Release ordering.
- A query captures the ready mask once with Acquire ordering.
- Lease validation and `installed_sequence` remain ingestion-worker-local.

Multi-bucket delta application is intentionally not atomic. A live query may observe a mixture of
old and new bucket images, and a query that captured readiness may finish after the lane is
retired. Do not add a lane seqlock, table-wide publication lock, double buffer, or query retry to
strengthen this documented weak-read contract.

## Lane recovery and failure boundaries

The consumer accepts a delta only when its identity and lease are current, its base sequence
equals the installed sequence, its next sequence is contiguous, and its bucket indices are valid
and unique. Stale or superseded traffic is ignored. A gap, malformed current-stream delta,
delivery uncertainty, lag-policy violation, or mid-application consumer failure retires only the
affected lane and requires a new lease plus barrier snapshot.

Recovery targets the state whose commit became uncertain:

- Suspect exact producer state requires a producer-generation rebuild.
- A trustworthy producer with an uncertain publication stream needs a new lease and snapshot.
- An uncertain consumer lane needs lane retirement and a new snapshot, not producer recovery.
- Ordinary pre-commit capacity exhaustion is an omission and requires no recovery.

The in-process `LocalCkfAdapter` exercises this complete boundary today. Non-local transport can
carry the same snapshot/delta protocol without moving exact full-hash ownership into the global
consumer.

## Directory map

| File | Responsibility |
| --- | --- |
| `addressing.rs` | Fingerprint and candidate-bucket derivation |
| `bucket.rs` | Packed producer buckets and atomic transposed consumer storage |
| `dc.rs` | Exact pool ownership, refcounts, mutations, dirty tracking, and snapshots |
| `mutator.rs` | Actor-owned insertion, relocation, rollback, and removal |
| `publication.rs` | Lease and sequence ownership for snapshots, deltas, and drains |
| `global.rs` | Manifest validation, lane state machine, transposed query storage |
| `ingestion_pool.rs` | Bounded lane-sticky Flume transport and readiness fencing |
| `adapter.rs` | In-process producer-to-consumer composition and snapshot recovery |
| `search.rs` | Multi-lane prefix search over the transposed table |
| `failure.rs` | Pre-/post-commit failure dispositions by affected domain |

## Correctness checks

Tests should compare exact ownership, refcounts, representation multiplicity, and
producer/consumer convergence. Byte-identical producer layouts and deterministic relocation paths
are not required. Stable ready state after an exact drain must reconstruct the producer bytes and
must not introduce false negatives for successfully tracked hashes.
