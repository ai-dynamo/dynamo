# KV Router Sharding — Design & Results

This document describes all sharding strategies implemented for the KV Router indexer,
their motivations, trade-offs, and benchmark results.

For the base data structures (`RadixTree`, `ConcurrentRadixTree`, `PositionalIndexer`)
see the [main indexer README](README.md).  For raw benchmark numbers and sweep data
see [`lib/kv-router/sharding_progress.md`](../../sharding_progress.md).

---

## Motivation

As Dynamo deployments grow to larger numbers of inference workers, the single-tree indexer
becomes a throughput bottleneck in two ways:

**KV event volume scales with worker count.** Each worker continuously emits `Stored` and
`Removed` events as it fills and evicts its KV cache.  A 100-worker deployment generates
~100× the event rate of a single worker.  All of these funnel into one shared
`ConcurrentRadixTreeCompressed` (CRTC), increasing write-lock contention on tree nodes
even though `ThreadPoolIndexer` serializes each worker's writes onto a dedicated OS thread.

**`find_matches` cost grows with worker count.** The CRTC traversal is O(D × W): at each
tree depth D it must track and intersect the set of candidate workers W.  More workers
means more set operations per query, so per-call latency rises as the deployment scales.
At high offered request rates, many `find_matches` calls compete for the same shared tokio
thread pool, compounding the slowdown.

Sharding partitions the prefix space across N independent CRTC instances so each shard
handles 1/N of workers and 1/N of event volume, keeping both costs bounded as the
cluster grows.

The key design tension is between:

- **Write routing** — events must land on the right shard or be broadcast (expensive).
- **Read routing** — `find_matches` should hit exactly one shard, not scatter-gather, or
  throughput does not scale.
- **Load balance** — shards must receive roughly equal load or one becomes the bottleneck.

---

## Roadmap

Sharding is being validated and deployed in three phases:

### Phase 1 — Single-process algorithm validation ✅

Implement and benchmark all sharding strategies within a single process, sharing
memory between the coordinator and shard trees.  This isolates the algorithmic
questions (routing correctness, load balance, early-exit rate) from deployment
concerns.

Work here covers `KvIndexerSharded`, `ShardedConcurrentIndexer`,
`PrefixShardedIndexer`, `BranchShardedIndexer`, `RebalancingBranchShardedIndexer`,
and `NodeDepthShardedIndexer` — all described in detail below.

### Phase 2 — Multi-process validation over UDS 🔄

Run each shard as a separate OS process, with the coordinator communicating over
Unix domain sockets (UDS).  This models a real deployment where each shard has a
dedicated CPU set and memory space.  UDS eliminates the network stack so IPC
overhead is a lower bound on what a same-host multi-process deployment can achieve.

The goal is to confirm that single-process algorithm wins (lower p99, better
balance) survive the process boundary and that the UDS overhead is acceptable
relative to CRTC traversal time.

**Current state:** The bench-side coordinator implementations exist in
`lib/bench/kv_router/` (`multi_process_sharded.rs`, `multi_process_node_depth.rs`,
`multi_process_node_depth_uds.rs`, `shard_server.rs`) but are **not wired into
`Cargo.toml`** and not declared as `mod` anywhere — they do not compile as-is.

**Next steps to complete Phase 2:**
1. Register `shard_server.rs` as a `[[bin]]` in `lib/bench/Cargo.toml` so it can be
   started as a standalone shard process.
2. Declare the coordinator modules in `mooncake_bench.rs` (or `common/mod.rs`) so
   they are compiled and accessible as bench subcommands.
3. Move the coordinator logic (`NodeDepthShardedIndexer` routing state) into
   `lib/kv-router/src/indexer/` so it is available as a proper indexer variant, not
   just bench-only code.
4. Wire the coordinator to the existing standalone indexer's event transport (ZMQ/NATS
   ingest + UDS query path) — see [`standalone_indexer_ipc.md`](../../standalone_indexer_ipc.md)
   for the serialization boundaries involved.

**Exit criteria:** UDS overhead stays below 10× in-process p99 at the target context
length, and shard balance on both mooncake and prefix traces matches the single-process
results.

### Phase 3 — Kubernetes deployment (future work) 🔲

Deploy the sharded router as a set of shard pods behind a coordinator pod in
Kubernetes.  Each shard pod runs its own CRTC with dedicated resources; the
coordinator holds the routing table (branch key → shard pod) and forwards KV events
and queries over the cluster network.

Key open questions: coordinator fault tolerance, shard discovery via k8s service
endpoints, and whether RDMA or kernel-bypass networking is needed to keep wire
overhead below CRTC traversal time at production context lengths.

---

## Sharding Strategy Comparison

| Indexer | `find_matches` | Event routing | Key challenge |
|---------|---------------|---------------|---------------|
| `KvIndexerSharded` | scatter-gather all shards | sticky by worker | O(N) query fanout |
| `ShardedConcurrentIndexer` | scatter-gather all shards | sticky by worker | spawn_blocking overhead |
| `PrefixShardedIndexer` | single shard (hash % N) | hash of first K blocks | hot-prefix skew |
| `BranchShardedIndexer` | single shard (routing table) | least-loaded assignment | needs known prefix depth |
| `RebalancingBranchShardedIndexer` | single shard (routing table) | least-loaded + background migrate | migration race window |
| `NodeDepthShardedIndexer` | single shard (shadow trie) | shadow trie walk | memory for shallow events |
| `MultiProcessShardedIndexer` | scatter-gather via HTTP | sticky by worker | loopback TCP overhead |
| `MultiProcessNodeDepthShardedIndexer` | single shard via HTTP/UDS | shadow trie + HTTP/UDS | IPC overhead |

---

## Phase 1 — Worker-Sharded (`KvIndexerSharded`, `ShardedConcurrentIndexer`)

**Files:** `sharded.rs`, `sharded_concurrent.rs`

Each worker is permanently assigned to one shard on first event (least-loaded by worker
count).  All KV blocks for that worker live in the assigned shard's tree.

**`find_matches`** must scatter to all N shards, because a query for an arbitrary prefix
does not know which worker holds it.  Results are gathered and merged.

### Result

`spawn_blocking` scatter-gather dominates latency: 77% overhead on 2 shards.

```
ShardedConcurrentIndexer find_matches:
  avg outer      = 3,455 µs
  avg max-shard  = 796 µs   (23% — pure CRTC work)
  avg overhead   = 2,659 µs (77% — spawn_blocking futex wake-up)
```

**Lesson:** scatter-gather does not scale.  Every additional shard multiplies query volume.
Single-shard routing is required for read throughput to scale with shard count.

---

## Phase 2 — Prefix-Sharded (`PrefixShardedIndexer`)

**File:** `prefix_sharded.rs`

Routes by `FNV(first prefix_depth block hashes) % N`.  Both events and queries use the
same hash, so `find_matches` always hits exactly one shard — no scatter-gather.

### Shard assignment

```
shard = FNV(blocks[0..min(prefix_depth, len)]) % N
```

Continuation events (`parent_hash = Some(h)`) inherit the shard via a `block_to_shard`
index keyed on the parent block hash, ensuring the entire conversation chain lives on one
shard.

### Problem: hot-prefix skew

If most requests share a long system prompt, all conversations hash to the same first
`prefix_depth` blocks → the same shard gets 100% of traffic.

```
prefix-sharded depth=1 — shard collapse:
  shard 0: 2,124,423 blocks (100%)
  shard 1:         0 blocks   (0%)
```

With `depth=2` the split improves to 69%/31% on the mooncake trace, still highly biased.

### Benchmark (mooncake trace, 2×4 workers)

| Depth | Peak throughput | vs baseline | p99 |
|-------|----------------|-------------|-----|
| 1 | collapses to single shard | — | — |
| 2 | 298k ops/s | **+37%** | 1,557 µs |

---

## Phase 3 — Branch-Sharded (`BranchShardedIndexer`)

**File:** `branch_sharded.rs`

Replaces `hash % N` with a **routing table** that assigns new branch keys to the
**least-loaded shard** at insertion time.  A "branch key" is `FNV(first prefix_depth
block hashes)` — but unlike prefix-sharding, the destination is chosen by load, not by
hash value.

### Key properties

- **Single-shard `find_matches`**: route to the branch's assigned shard, or early-exit if
  the branch key is unknown (no worker has ever stored that prefix).
- **Least-loaded assignment**: measured by live block count (O(1) atomic read); branch
  count used as tiebreaker at startup.
- **Stable assignment**: once assigned, a branch never migrates.  CRTC-internal splits
  stay within the owning shard.
- **Unknown-branch fast path**: unrecognized branch key → return empty scores in ~300 ns
  without dispatching to any shard.

### FNV accumulation for shallow chains

Conversations shorter than `prefix_depth` blocks carry a partial FNV accumulator forward
via `block_to_fnv_state`.  When a continuation extends the chain to `prefix_depth`, the
finalized key assigns the branch — each distinct conversation gets its own shard even
when all conversations share a long common prefix.

### Remove routing

1. **Mapped (primary)**: `block_to_shard` index routes Remove to the owning shard.
2. **Broadcast fallback**: blocks absent from the index (evicted/OOO) are broadcast; each
   shard treats a missing block as a no-op.

### Benchmark (mooncake trace, 2×4 workers)

```
BranchShardedIndexer find_matches (165,256 total):
  99,976 dispatched, 65,280 early-exit (39.5% miss)
  avg routing = 300 ns  (routing table lookup)
  avg shard   = 858 µs  (CRTC traversal)
  shard 0: 673,927 blocks (50.6%)
  shard 1: 657,341 blocks (49.4%)
```

| Config | Peak throughput | vs baseline | p99 |
|--------|----------------|-------------|-----|
| CRTC baseline (8w) | 218k ops/s | — | 1,941 µs |
| Branch-sharded 2×4w, depth=2 | **376k ops/s** | **+73%** | 1,795 µs |

**Why it beats prefix-sharded by 26%:** the 39.5% early-exit rate.  Those calls complete
in ~300 ns instead of ~858 µs, so effective average call time is:

```
branch-sharded: 0.395 × 300ns + 0.605 × 858µs ≈ 520 µs
prefix-sharded: 1.000 × 795µs               ≈ 795 µs
→ branch-sharded ~35% lower average → ~54% higher throughput
```

### Limitation: depth must be tuned per workload

On a conversational trace with a 15-block system prompt, `depth=2` means all conversations
share the same 2-block prefix → only one branch key → one shard gets everything.
`depth=17` (prompt length + 2) produces distinct keys per conversation, near-perfect
balance, and an 83% early-exit rate.  But that requires knowing the prompt length in
advance.

---

## Phase 4 — Rebalancing (`RebalancingBranchShardedIndexer`)

**File:** `branch_sharded_rebalancing.rs` — see [`REBALANCING.md`](REBALANCING.md) for the full design and protocol details.

Extends `BranchShardedIndexer` with a background task that detects hot shards and
migrates the hottest branch to the coolest shard.

### Two-phase migration protocol

A naive "dump → replay → switch" approach has a race: events arriving between the dump
and the routing switch land only on the old shard.  The fix uses dual-write + scatter-
gather during migration:

```
Phase 1 — Replaying
  Events     → old shard only
  find_matches → old shard only
  Migration task: dump old shard, replay Stored events into new shard's FIFO queue

Phase 2 — DualWrite  (activated atomically after replay enqueue)
  Events     → old shard AND new shard
  find_matches → scatter-gather both, merge OverlapScores
  Migration task: wait for queues to drain, switch routing, exit DualWrite
```

FIFO ordering via `flume` channels guarantees replay events are processed before any
subsequent dual-write events for the same worker.

### Benchmark

The rebalancer fires idle on the mooncake trace (no hot shard detected), adding ~4%
overhead from the background polling.  Improvement over BranchShardedIndexer is marginal
at this load level but the migration correctness guarantee is preserved.

---

## Phase 5 — Node-Depth Shadow-Trie (`NodeDepthShardedIndexer`)

**File:** `node_depth_sharded.rs` — see [`NODE_DEPTH_SHARDING.md`](NODE_DEPTH_SHARDING.md) for the full design, the imbalance/incoherence trade-off, and multi-shard scaling results.

Solves the depth-tuning problem of `BranchShardedIndexer` by routing on the first
`routing_node_depth` **CRTC nodes** traversed, not on raw block count.

### Why node depth instead of block depth

Because the CRTC path-compresses shared prefixes into single edges, a 15-block system
prompt becomes **one node**.  At `routing_node_depth=2`, routing already sees all distinct
conversation continuations without knowing the prompt length.

### Shadow trie

A `ShadowNode` trie mirrors the CRTC's top-`routing_node_depth` node structure.  Each
edge is a `Vec<LocalBlockHash>` (variable length — exactly one CRTC-node's worth of
blocks).

- **Routing**: read-lock the trie, walk `routing_node_depth` edges → return shard.
  O(routing_node_depth) hash lookups.
- **Insertion**: write-lock the trie, walk/create/split.  Only root events (or all events
  when `inherit_parent_shard=false`).  Continuation events fast-path via `block_to_shard`.

### Hybrid depth-aware routing

`last_block_to_path` tracks the full prefix for each shallow event so that incremental
blocks are inserted as proper children, not phantom siblings:

- **Parent reached a routing leaf** (`last_block_to_path = None`): inherit shard.
  No trie access needed; correct forever.
- **Parent is shallow** (`last_block_to_path = Some(prefix)`): reconstruct full sequence
  and re-walk the trie.
- **Parent not tracked** (root event / OOO / evicted): insert incremental blocks from
  root.

### Benchmark (prefix trace, 4×4 workers)

| Config | p99 latency | avg routing | avg shard |
|--------|------------|-------------|-----------|
| CRTC 16 workers | 254 µs | — | — |
| NodeDepthSharded 4×4, depth=2 | **87 µs** | 1,607 ns | 94 µs |

83% early-exit rate; perfect shard balance (50%/50%).

---

## Phase 6 — Multi-Process Sharded (`MultiProcessShardedIndexer`)

**File:** `lib/bench/kv_router/multi_process_sharded.rs`

Coordinator routes events and queries to separate OS processes over loopback TCP HTTP.
Events are sticky-routed by worker; `find_matches` scatter-gathers to all shard processes.

### Purpose

Measures the overhead of the HTTP/JSON serialization layer to bound what a real
multi-node deployment (with dedicated CPUs per shard) would cost.

### Benchmark (mooncake trace, 2 shards)

```
MultiProcessShardedIndexer find_matches (2 shards):
  avg outer         = 115,677 µs
  CRTC traversal   =     597 µs  (0.5%)
  JSON total        =   1,094 µs  (0.9%)
  network (loopback)= 113,655 µs (98.3%)
```

**Conclusion:** loopback TCP RTT dominates.  Multi-process sharding is only competitive
when shards run on separate hosts where CRTC traversal is the dominant cost (very long
context windows, O(10ms) traversal time).

---

## Phase 7 — Multi-Process Node-Depth (`MultiProcessNodeDepthShardedIndexer`)

**Files:** `lib/bench/kv_router/multi_process_node_depth.rs`,
`lib/bench/kv_router/multi_process_node_depth_uds.rs`

Same as Phase 6 but routes `find_matches` to **exactly one shard** using the coordinator's
shadow trie.  Tested over both loopback TCP and Unix domain sockets (UDS).

### Benchmark (prefix trace, 2 shards, 37k ops/s)

| Transport | avg outer | p99 | Wire overhead | CRTC (%) |
|-----------|----------|-----|--------------|---------|
| TCP loopback | 5,027 µs | 5,428 µs | 4,688 µs (93%) | 113 µs (2.2%) |
| UDS | **1,645 µs** | **877 µs** | 1,264 µs (77%) | 128 µs (7.8%) |
| In-process | ~96 µs | 135 µs | — | 94 µs (98%) |

UDS is **6.2× lower p99** than TCP; in-process is still **6× lower p99** than UDS.

**Why UDS is faster:** Unix domain sockets bypass the full TCP stack (no ACKs,
congestion control, virtual NIC ring buffers).  The residual ~1ms UDS overhead is
tokio task scheduling + axum HTTP parsing — would require a raw binary protocol to
eliminate.

### 1-shard baseline vs 2-shard node-depth (TCP, same offered load)

| | 1 shard (baseline) | 2 shards (node-depth) |
|--|--------------------|-----------------------|
| CRTC traversal | 174 µs | **113 µs (−35%)** |
| avg outer | 5,169 µs | 5,027 µs (≈ same) |
| JSON deser (coord) | 342 µs | **178 µs (2× faster)** |

Sharding does reduce CRTC traversal by 35% (smaller trees), but the network RTT is
identical, so end-to-end latency is unchanged at this load level.

---

## Summary of Results

### In-process, single-machine (mooncake trace, 2×4 workers)

| Indexer | Peak ops/s | vs baseline | p99 |
|---------|-----------|-------------|-----|
| CRTC baseline (8w) | 218k | — | 1,941 µs |
| Prefix-sharded depth=2 | 298k | +37% | 1,557 µs |
| Branch-sharded depth=2 | **376k** | **+73%** | 1,795 µs |

### In-process, prefix conversational trace (4×4 workers)

| Indexer | p99 | Notes |
|---------|-----|-------|
| CRTC 16 workers | 254 µs | single tree, read contention |
| Branch-sharded 4×4, depth=17 | **84 µs** | 83% early-exit |
| Node-depth 4×4, depth=2 | **87 µs** | no prompt-length knowledge needed |

### Multi-process (prefix trace, 2 shards, 37k ops/s)

| Transport | p99 |
|-----------|-----|
| TCP loopback | 5,428 µs |
| UDS | 877 µs |
| In-process | **135 µs** |

---

## Running the Benchmarks

All sharding benchmarks run through `mooncake_bench` in `dynamo-bench`:

```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  <TRACE_PATH> [global options] <INDEXER_SUBCOMMAND> [indexer options]
```

Two traces are used throughout:

- **mooncake** (`lib/kv-router/mooncake_trace.jsonl`) — production trace with diverse prefixes
- **prefix** (`.prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl`) — synthetic
  conversational trace with a 15-block shared system prompt; use this to stress-test shared-prefix
  balance

### Example: branch-sharded vs CRTC baseline

```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  lib/kv-router/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

Key arguments:

| Argument | What it does |
|----------|-------------|
| `--trace-simulation-duration-ms` | How long of the trace to replay as KV events before measuring |
| `--benchmark-duration-ms` | Measurement window; shorter = higher offered rate = more stress |
| `-d` | Number of concurrent `find_matches` senders; higher stresses reads more |
| `--num-shards` | Number of independent CRTC shards |
| `--num-event-workers-per-shard` | OS threads handling KV events on each shard |
| `--prefix-depth` | Number of blocks hashed to compute the branch key |

### Example: side-by-side comparison and throughput sweep

Use `--compare` to run multiple indexers against the same trace replay:

```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  .prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl \
  --num-event-workers 8 -d 3 --benchmark-duration-ms 30000 \
  --compare concurrent-radix-tree-compressed,branch-sharded-crtc,node-depth-sharded-crtc
```

To find the saturation point of an indexer, add sweep flags — the bench varies
`--benchmark-duration-ms` across steps, with shorter windows producing higher offered rates:

```bash
  --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8
```

---

## Known Issues

### Block skew in `BranchShardedIndexer` on production traces

On the mooncake trace, `BranchShardedIndexer` at `depth=2` produces perfect **branch**
balance (1,749 branches per shard) but severely skewed **block** balance (shard[3] holds
38% of blocks vs 17% on shards[1,2]).  The root cause is that least-loaded assignment
counts branches, not blocks — some branches accumulate far more blocks than others (high-
reuse long conversations).  The hot shard's larger tree drives up its p99, eliminating
the latency advantage over CRTC.

**Proposed fix:** weight shard assignment by live block count instead of branch count.
`BranchShardedIndexer` already tracks `shard_block_counts` as `AtomicUsize` per shard
(used for metrics); switching the assignment comparator from branch count to block count
is a small change.  This has not yet been benchmarked.

### Rebalancer cannot fix shared-prefix skew

`RebalancingBranchShardedIndexer` at `depth=2` on the prefix trace enters infinite
oscillation: the 15-block shared system prompt produces exactly one branch key, so the
rebalancer migrates it to the other shard, then immediately detects imbalance again and
migrates back.  During each dual-write window p99 nearly doubles.

The rebalancer cannot subdivide a single branch key — it has no visibility into the
finer-grained branching below `prefix_depth`.  The correct fix is `NodeDepthShardedIndexer`,
which routes at CRTC node granularity and handles this case structurally.

### Multi-process files are not wired up

The Phase 6/7 coordinator and shard server files in `lib/bench/kv_router/`
(`multi_process_sharded.rs`, `multi_process_node_depth.rs`, `multi_process_node_depth_uds.rs`,
`shard_server.rs`) are not compiled — they were included for reference but left
unwired to keep ai-dynamo/dynamo#7859 focused.  See the Phase 2 roadmap entry
above for the steps needed to wire them up.

Note: the UDS transport is already fully implemented in `multi_process_node_depth_uds.rs`
and benchmarked (see Phase 7 results above) — it is not future work, just not yet wired
into the build.

---

## Choosing a Sharding Strategy

| Workload | Recommended | Why |
|----------|-------------|-----|
| Conversational (long shared system prompt, distinct conversations) | `NodeDepthShardedIndexer` | No need to know prompt length; 83%+ early-exit |
| Conversational with known prompt length | `BranchShardedIndexer` | Simpler, same performance as node-depth |
| Diverse workloads (no dominant shared prefix) | `PrefixShardedIndexer` | Simple, near-uniform load |
| Shards on separate hosts (RDMA/NIC) | `MultiProcessNodeDepthShardedIndexer` via UDS | IPC overhead <1% of CRTC traversal time for long contexts |
| Debugging / scatter-gather baseline | `ShardedConcurrentIndexer` | Not recommended for production |
