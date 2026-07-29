# Onboarding Guide

Welcome to `kvbm-engine`. This document walks you through the core abstractions
in the crate so you can orient yourself quickly and start contributing.

`kvbm-engine` is the distributed coordination layer for KV cache block management
(KVBM). It sits above `kvbm-physical` (which moves bytes) and `kvbm-logical`
(which tracks block metadata), stitching them together into a system where
**leaders** make decisions about blocks and **workers** execute data transfers
across a tiered storage hierarchy:

```text
G1 (GPU HBM)  →  G2 (Pinned DRAM)  →  G3 (NVMe/SSD)  →  G4 (S3/MinIO)
```

The central design tension is between **logical** and **physical**. Leaders think
in sequence hashes and block identities — they never touch raw memory. Workers
think in layout handles, transfer managers, and DMA descriptors — they never make
placement decisions. The engine holds these two worlds together.

---

## The Worker

A worker is the physical side of the logical-physical dichotomy. The core
implementation is `PhysicalWorker`, a thin coordination wrapper around
`kvbm-physical`.

A `PhysicalWorker` owns:

- A **`TransferManager`** — the `kvbm-physical` engine that actually moves data
  between memory regions via NIXL (RDMA/UCX), NVMe, or object storage APIs.
- **Layout handles** for up to three tiers (`g1_handle`, `g2_handle`,
  `g3_handle`) — these are physical memory region registrations that the transfer
  manager uses to know *where* data lives on this process.
- A map of **remote handles** — physical handles imported from peer workers,
  enabling RDMA pulls.

Workers implement two traits:

**`WorkerTransfers`** defines the transfer operations:
- `execute_local_transfer(src, dst, block_ids, ...)` — move blocks between tiers
  within this worker (e.g. G2 → G1).
- `execute_remote_onboard(remote_desc, dst, block_ids, ...)` — RDMA pull from a
  remote worker into a local layout.
- `execute_remote_offload(src, remote_desc, block_ids, ...)` — push local data
  to a remote descriptor.
- `connect_remote(instance_id, metadata)` — import a peer's NIXL metadata so we
  can do RDMA to/from them.

**`Worker`** extends `WorkerTransfers` with layout handle accessors and metadata
import/export for RDMA setup.

All transfer operations return a `TransferCompleteNotification` — an async handle
you await to know when the data movement is done. This is how the system achieves
overlap between transfer scheduling and transfer execution.

---

## Workers as Remote Services (Velo)

In a multi-process deployment, each worker runs in its own process. Rather than
calling `PhysicalWorker` methods directly, we wrap it as a Velo RPC service.

**`VeloWorkerService`** takes a `PhysicalWorker` and registers handlers for
every `WorkerTransfers` and `Worker` method (e.g. `kvbm.worker.local_transfer`,
`kvbm.worker.remote_onboard`, etc.). The service lives in the worker process.

**`VeloWorkerClient`** implements the same `Worker` trait but serializes each
call into a Velo message, sends it to the remote service, and returns a
`TransferCompleteNotification` backed by a completion event.

The key insight: **from the leader's perspective, local and remote workers are
interchangeable.** Both implement `Worker`. The leader never knows (or cares)
whether it is talking to an in-process `PhysicalWorker` or a `VeloWorkerClient`
that crosses a process boundary.

```text
Leader process                          Worker process
┌───────────────────┐                   ┌───────────────────┐
│  InstanceLeader   │                   │                   │
│        │          │                   │                   │
│  CoordinatedWorker│                   │                   │
│        │          │                   │                   │
│  VeloWorkerClient │ ── Velo RPC ──▶  │ VeloWorkerService │
│                   │                   │        │          │
│                   │                   │  PhysicalWorker   │
│                   │                   │  (TransferManager)│
└───────────────────┘                   └───────────────────┘
```

There is one more wrapper to mention: **`CoordinatedWorker`**. This lives in the
leader process and adds coordination state on top of a `Worker` (local or
remote). It tracks the leader's view of which layout handles map to which
remote instances and ranks. When the leader says "pull blocks from Instance B,
rank 0", the `CoordinatedWorker` resolves the correct physical handle and
delegates to the inner `Worker`.

---

## Worker Groups

Workers can be organized into groups that present a single-worker interface to
the leader. The `ParallelWorkers` trait is the group-level analog of `Worker`.

### Tensor Parallel (SPMD)

`SpmdParallelWorkers` is the default group implementation. It broadcasts every
operation to all N workers in parallel — the SPMD (Single Program, Multiple Data)
model.

In a typical tensor-parallel deployment, each GPU holds its own shard of every
KV cache block. When the leader says "transfer blocks [1, 2, 3] from G2 to G1",
the SPMD group fans this out to every rank. Each rank executes the same transfer
on its own shard. Results are aggregated before returning to the leader.

```text
Leader: "transfer blocks 1,2,3 from G2 → G1"
         │
   SpmdParallelWorkers
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Rank0 Rank1 Rank2    (each transfers its own shard)
```

### Replicated Data (MLA)

For Multi-head Latent Attention (MLA), KV data is replicated rather than sharded.
The `ReplicatedDataWorker` (feature-gated behind `collectives`) implements a
different strategy:

- **Rank 0** is the only worker with G2 and G3 storage. It performs all
  tier-to-tier transfers (G3 → G2 → G1).
- **Ranks 1..N** only have G1. They receive data from rank 0 via NCCL
  `broadcast`.

This means the leader can still say "onboard these blocks" and the group handles
the asymmetry internally — rank 0 does the heavy lifting, then broadcasts to
everyone else.

### The Power of the Abstraction

These two strategies — symmetric sharding and replicated broadcast — are very
different physically, but the leader drives both through the same
`ParallelWorkers` / `WorkerTransfers` interface. This is the core value of worker
groups: **different parallelism strategies behind a uniform API**.

The abstraction is admittedly incomplete — more parallelism patterns will need
more group implementations — but it is sufficient for the two use cases presented
and demonstrates the pattern for extending it.

---

## The Leader

The leader is the logical counterpart to the worker. `InstanceLeader` owns the
logical view of all block data, regardless of how it is physically distributed
across workers and tiers.

An `InstanceLeader` holds:

- A **`BlockRegistry`** for deduplication — tracks which sequence hashes have
  been seen.
- A **`BlockManager<G2>`** (required) and optional **`BlockManager<G3>`** — the
  logical block stores for host DRAM and disk.
- A list of **workers** (via `CoordinatedWorker`) and an optional
  **`SpmdParallelWorkers`** group.
- A map of **sessions** for distributed onboarding (more on this below).
- Optional **remote leader** references for cross-instance coordination.

### find_matches

The core entry point is `find_matches(sequence_hashes)`. Given a list of
sequence hashes, the leader determines which blocks already exist and where:

1. Search the local G2 `BlockManager` for matches.
2. Search the local G3 `BlockManager` for any remaining hashes.
3. Optionally search remote leaders via distributed sessions.

The result is either:
- **`Ready`** — all requested blocks were found locally in G2; the caller gets
  immediate RAII `BlockHolder`s.
- **`AsyncSession`** — some blocks require staging (G3 → G2) or remote transfers;
  the caller gets a session handle with a status watch channel.

### BlockHolder (RAII Ownership)

`BlockHolder<T>` (where T is `G2` or `G3`) is an RAII guard that holds blocks
during a session. While held, those blocks cannot be evicted. When the holder is
dropped, blocks are released. This prevents leaks even if session handling
panics.

### Block Scanning

`InstanceLeader` also exposes `scan_with_policy` — a flexible iteration
mechanism where the caller provides a closure that searches for blocks using a
`BlockAccessor` (which wraps both G2 and G3 managers) and yields results through
a `PolicyContext`. This enables custom scanning strategies (contiguous runs,
LFU-sorted scans) without exposing block manager internals.

---

## Instances

An **Instance** is the deployment unit: one leader plus its workers.

```text
┌─ Instance (TP=2) ──────────────────────────┐
│                                             │
│   InstanceLeader                            │
│       │                                     │
│   SpmdParallelWorkers                       │
│       ├── Worker (rank 0, GPU 0)            │
│       └── Worker (rank 1, GPU 1)            │
│                                             │
└─────────────────────────────────────────────┘
```

In a single-GPU setup, the instance is simply one leader and one worker.
In tensor-parallel, it is one leader driving an SPMD group.

The leader drives; the workers execute. The leader never touches bytes; the
workers never make placement decisions.

---

## Transfer Classification

Transfers fall into three classes based on scope:

### Local (intra-worker, intra-instance)

Tier-to-tier transfers within a single worker: G1 ↔ G2, G2 ↔ G3, etc.

This is the bread and butter of a tensor-parallel deployment. Each worker
independently moves its own shard between tiers. The SPMD group broadcasts the
same logical operation to all ranks, and each rank executes it on its own
physical layouts.

### Intra (inter-worker, intra-instance)

Transfers between workers within the same instance. The motivating example is the
MLA/replicated data pattern: rank 0 performs a G3 → G2 → G1 transfer, then
NCCL broadcasts its G1 data to all other ranks. The data crosses worker
boundaries but stays within the same instance.

### Inter (inter-worker, inter-instance)

Transfers between workers on different instances. This is **distributed KVBM** —
the peer-to-peer model described in the next section.

```text
           ┌──────────────────────────────┐
           │          Local               │
           │    (intra-worker, intra-inst) │
           │     G2 ←→ G1 on Rank 0      │
           └──────────────────────────────┘

  ┌──────────────────────────────────────────────┐
  │              Intra                            │
  │       (inter-worker, intra-inst)              │
  │   Rank 0 ──NCCL bcast──▶ Rank 1..N           │
  └──────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │                       Inter                                  │
  │              (inter-worker, inter-inst)                       │
  │   Instance A, Rank 0 ──RDMA──▶ Instance B, Rank 0           │
  └──────────────────────────────────────────────────────────────┘
```

---

## Distributed KVBM (Inter-Instance Transfers)

Distributed KVBM is a peer-to-peer model where two or more instances coordinate
block ownership through **sessions**, then trigger direct worker-to-worker
transfers.

### Sessions

A session is a short-lived coordination protocol between two instances. There are
two roles:

- **`InitiatorSession`** — the requesting side (e.g. a Prefill instance that
  needs blocks).
- **`ResponderSession`** — the providing side (e.g. a Decode instance that has
  blocks cached).

Sessions progress through a state machine:

```text
Searching ──▶ Holding ──▶ Staging ──▶ Ready ──▶ Complete
                                                  │
                                             (or Failed)
```

- **Searching**: The initiator asks the responder to search its local block
  managers.
- **Holding**: The responder has found blocks and holds them via `BlockHolder` to
  prevent eviction.
- **Staging**: G3 → G2 promotion is in progress on the responder (if blocks were
  on disk). NIXL descriptors are prepared for RDMA.
- **Ready**: Blocks are in G2 on the responder and RDMA-ready.
- **Complete**: The initiator has pulled all blocks. The session is torn down.

### Worked Example: TP=2 Cross-Instance Transfer

Suppose Instance A (Prefill, TP=2) wants KV blocks for sequence hashes
`[hash_1, hash_2]` from Instance B (Decode, TP=2).

```text
Instance A (Prefill, TP=2)              Instance B (Decode, TP=2)
┌──────────────────────┐                ┌──────────────────────┐
│ Leader A             │                │ Leader B             │
│  ├─ Worker A0 (GPU0) │                │  ├─ Worker B0 (GPU0) │
│  └─ Worker A1 (GPU1) │                │  └─ Worker B1 (GPU1) │
└──────────────────────┘                └──────────────────────┘
```

The flow:

1. **Leader A creates a session** with Leader B, sending the sequence hashes
   `[hash_1, hash_2]` it is looking for.

2. **Leader B receives the request** (`ResponderSession`). It searches its G2
   and G3 block managers for matches.

3. **Leader B acquires ownership** of the matched blocks via `BlockHolder`,
   preventing eviction during the transfer.

4. **Leader B responds** with what it found: which hashes matched, their
   tier locations, and NIXL descriptors that allow RDMA access to the G2 blocks.

5. **Leader A instructs its workers to pull.** Since both instances use TP=2, the
   mapping is 1:1 — rank 0 on A pulls from rank 0 on B, rank 1 on A pulls from
   rank 1 on B. Each pull is a direct RDMA transfer between the worker processes
   using NIXL.

6. **Session completes.** Leader B releases its `BlockHolder`s. Leader A now has
   the blocks in its own G2.

The rank mapping is handled by `route_local_to_remote` in `LeaderState`, which
supports asymmetric configurations too (e.g. TP=4 pulling from TP=2).

### Transport

Session messages travel over **Velo** (the project's RPC framework).
`VeloLeaderService` registers handlers for `kvbm.leader.onboard`,
`kvbm.leader.remote_session`, and `kvbm.leader.session` — these dispatch
incoming messages to the appropriate per-session channels.

For testing, `LocalTransport` provides direct in-process dispatch without
network overhead.

---

## Objects vs Blocks

Throughout the crate, you will encounter two distinct representations of
KV cache data:

### Blocks

A **Block** is the fundamental unit within tiers G1–G3. It is identified by a
`BlockId`, associated with a `SequenceHash`, and managed by a `BlockManager`.
Blocks have physical backing (GPU HBM, pinned DRAM, or NVMe) and support
direct memory transfers via NIXL. The `BlockManager` handles allocation,
eviction, and frequency tracking. Blocks are the hot-path, low-latency
representation.

### Objects

An **Object** is the G4 (S3/MinIO) representation. Objects are addressed by
**key** (derived from a `SequenceHash` via a `KeyFormatter`), not by `BlockId`.
The `ObjectBlockOps` trait defines the interface: `has_blocks`, `put_blocks`,
`get_blocks`.

Objects exist because S3 does not support the block-oriented, handle-based access
pattern of the lower tiers. They provide unlimited-capacity cold storage at the
cost of higher latency and a key-value access model.

For SPMD deployments, the `RankPrefixedKeyFormatter` prefixes each object key
with the worker rank (`{rank}/{hash}`), so each worker's shard is stored
independently.

The `ObjectLockManager` provides distributed locking for G4 writes using
conditional S3 PUTs, preventing duplicate uploads across concurrent instances.

---

## Where to Go Next

Now that you have the conceptual model, dive into the per-module documentation
for implementation details:

| Document | Covers |
|----------|--------|
| [architecture.md](architecture.md) | Tier model, module map, feature flags, quick start |
| [leader.md](leader.md) | `Leader` trait, `InstanceLeader`, `FindMatchesResult`, staging modes |
| [worker.md](worker.md) | `Worker` / `WorkerTransfers`, `PhysicalWorker`, `CoordinatedWorker`, Velo layer |
| [worker-group.md](worker-group.md) | `SpmdParallelWorkers`, fan-out, rank-aware routing |
| [session.md](session.md) | Session protocol, initiator/responder/controllable, message types, state machine |
| [offload.md](offload.md) | Offload pipeline stages, policies, cancellation |
| [object.md](object.md) | G4 storage, S3 client, lock manager |
| [runtime.md](runtime.md) | `KvbmRuntime` construction and shared infrastructure |
| [testing.md](testing.md) | Test utilities, multi-instance fixtures, RDMA transfer tests |

To run the test suite:

```bash
cargo test -p kvbm-engine --features testing
```
