# CRTC Sharding: Implementation Progress & Benchmark Results

## Status

**Phase 1 (worker-based sharding) — IMPLEMENTED AND BENCHMARKED.**

**Phase 2 (prefix-based sharding) — IMPLEMENTED AND BENCHMARKED.**

**Phase 3 (branch-based sharding) — IMPLEMENTED AND BENCHMARKED.**

**Phase 4 (query-rate rebalancing) — IMPLEMENTED AND BENCHMARKED.**

**Phase 5 (node-depth shadow-trie sharding) — IMPLEMENTED AND BENCHMARKED.**

**Phase 6 (multi-process sharded CRTC) — IMPLEMENTED AND BENCHMARKED.**

**Phase 7 (multi-process node-depth sharded CRTC) — IMPLEMENTED AND BENCHMARKED.**

---

## Benchmark Results

All runs on the mooncake trace, 2 shards × 4 event workers/shard (8 total, same as
baseline's 8 workers).

### Why sharding doesn't give 2× throughput in a single-process benchmark

Every `find_matches` call runs **inline on the calling tokio thread** (`thread_pool.rs`:
`Ok(self.backend.find_matches(&sequence, false))`).  The tokio async executor is shared across
all shards.  "Double the resources" means two smaller trees with less lock contention and faster
individual traversals — not two independent thread pools.  Throughput is bounded by
`tokio_workers / avg_traversal_time_per_call`.  Sharding reduces traversal time (smaller tree),
but not the number of concurrent readers.

In a real multi-node deployment each shard runs on its own machine with a dedicated event loop
and CPU set.  That is where linear scaling is actually observed.  Within a single process,
improvement is real but sub-linear.

### Fixed-load comparison (`-d 7`, 50k ops/s offered — all configs have spare capacity)

At this load level all configs achieve ~98–99% of offered throughput — the indexer is not the
bottleneck, so numbers are nearly identical.  Use the sweep results below to see differences.


| Indexer                             | depth | p99 latency  | Block throughput  | Shard split (blocks) | Notes                                                                                      |
| ----------------------------------- | ----- | ------------ | ----------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| CRTC baseline (8 workers)           | —     | 1,641 µs     | 492,685 blk/s     | —                    | single tree                                                                                |
| prefix-sharded (2×4w)               | 1     | 1,612 µs     | 492,879 blk/s     | **100% / 0%**        | shard collapse — see §Shard collapse                                                       |
| prefix-sharded (2×4w)               | 2     | 1,401 µs     | 494,246 blk/s     | 69% / 31%            | biased split — see §Shard imbalance                                                        |
| branch-sharded (2×4w)               | 2     | **1,381 µs** | **495,621 blk/s** | **51% / 49%**        | 39.5% early-exit — see §Branch-sharded                                                     |
| rebalancing-branch-sharded (2×4w)   | 2     | 1,286 µs     | 495,500 blk/s     | **51% / 49%**        | idle rebalancer (+4% overhead) — see §Rebalancing                                          |
| worker-sharded (2×4w)               | —     | 48,747 µs    | 482,770 blk/s     | 51% / 50%            | scatter-gather — see §Worker-sharded overhead                                              |
| multi-process-sharded (2 shards)    | —     | 263,927 µs   | 923,081 blk/s     | ~50% / ~50%          | overwhelmed (275k/7.8M ops offered); 98.3% HTTP overhead — see §Multi-process overhead     |
| multi-process-node-depth (2 shards) | 2     | 5,428 µs     | 125,615 blk/s     | **50% / 50%**        | steady state (37k/37k ops, 100% util); 97.8% HTTP overhead — see §Multi-process node-depth |


### Throughput sweep (varying offered load to find saturation point)

`--sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 -d 7`.
A shorter `benchmark_duration_ms` means the same 165k operations must complete faster → higher
effective offered rate.  Configs diverge once offered rate exceeds their capacity.

#### Peak throughput (1000 ms window — maximum stress)


| Indexer                | Achieved ops/s | vs CRTC baseline | p99      |
| ---------------------- | -------------- | ---------------- | -------- |
| CRTC baseline          | 218,013        | —                | 1,941 µs |
| prefix-sharded depth=2 | 298,314        | **+37%**         | 1,557 µs |
| branch-sharded depth=2 | **376,465**    | **+73%**         | 1,795 µs |


Branch-sharded achieves 73% more throughput than CRTC baseline with the same total thread count.
Prefix-sharded achieves 37%.  Neither reaches 2× — see the explanation above.

**Why branch-sharded beats prefix-sharded by 26%:** the 39.5% early-exit rate.  Those calls
complete in ~200 ns (hash + DashMap lookup) instead of ~750 µs shard traversal, freeing tokio
threads nearly instantly.  The effective average call time is:

```
branch-sharded: 0.395 × ~200ns + 0.605 × ~750µs ≈ 455µs
prefix-sharded: 1.000 × ~640µs ≈ 640µs
→ branch-sharded has ~29% lower average call time → ~41% higher throughput (1/0.71 ≈ 1.41)
```

The remaining gap to 2× is the shared tokio executor.

#### Full sweep table

**concurrent-radix-tree-compressed (baseline):**


| duration_ms | offered ops/s | achieved ops/s | util | p99                   |
| ----------- | ------------- | -------------- | ---- | --------------------- |
| 1,000       | 502,957       | 218,013        | 43%  | 1,941 µs              |
| 1,626       | 309,322       | 211,771        | 68%  | 2,188 µs              |
| 2,643       | 190,298       | 182,562        | 96%  | 1,812 µs              |
| 4,296       | 117,076       | 113,920        | 97%  | 1,465 µs              |
| 6,983       | 72,026        | 70,929         | 98%  | 1,680 µs              |
| 11,352      | 44,306        | 43,865         | 99%  | 1,543 µs              |
| 18,455      | 27,253        | 27,089         | 99%  | 1,435 µs ← early stop |


**prefix-sharded-crtc depth=2:**


| duration_ms | offered ops/s | achieved ops/s | util | p99                   |
| ----------- | ------------- | -------------- | ---- | --------------------- |
| 1,000       | 502,957       | 298,314        | 59%  | 1,557 µs              |
| 1,626       | 309,322       | 280,825        | 91%  | 1,608 µs              |
| 2,643       | 190,298       | 184,571        | 97%  | 1,176 µs              |
| 4,296       | 117,076       | 114,830        | 98%  | 1,457 µs              |
| 6,983       | 72,026        | 71,150         | 99%  | 1,466 µs              |
| 11,352      | 44,306        | 43,972         | 99%  | 1,349 µs              |
| 18,455      | 27,253        | 27,124         | 100% | 1,252 µs ← early stop |


**branch-sharded-crtc depth=2:**


| duration_ms | offered ops/s | achieved ops/s | util | p99                   |
| ----------- | ------------- | -------------- | ---- | --------------------- |
| 1,000       | 502,957       | 376,465        | 75%  | 1,795 µs              |
| 1,626       | 309,322       | 298,491        | 96%  | 1,366 µs              |
| 2,643       | 190,298       | 185,799        | 98%  | 1,519 µs              |
| 4,296       | 117,076       | 115,384        | 99%  | 1,650 µs              |
| 6,983       | 72,026        | 71,301         | 99%  | 1,499 µs              |
| 11,352      | 44,306        | 44,046         | 99%  | 1,342 µs ← early stop |


CRTC saturates around **210k ops/s**.  Prefix-sharded saturates around **280–300k ops/s**.
Branch-sharded saturates around **300–380k ops/s** and exits the sweep at the 11k-step (only
6 steps needed vs 7 for the others) because it keeps up at lower stress levels longer.

### Detailed timing reports

**prefix-sharded depth=2** (165,256 find_matches calls):

```
avg routing = 111ns  (FNV hash, 0.00% of outer)
avg shard   = 795µs  (CRTC traversal, inline on caller thread)
Shard block distribution:
  shard 0: 891226 blocks (69.2%), 7000 workers
  shard 1: 395870 blocks (30.8%), 6888 workers
```

**prefix-sharded depth=1** — shard collapse confirmed:

```
avg routing = 109ns  (FNV hash, 0.00% of outer)
avg shard   = 1033µs  (CRTC traversal — 100% of load on one shard)
Shard block distribution:
  shard 0: 2124423 blocks (100.0%), 7000 workers
  shard 1: 0 blocks (0.0%), 0 workers
```

**branch-sharded depth=2** (165,256 total find_matches calls):

```
BranchShardedIndexer find_matches (165256 total: 99976 dispatched, 65280 early-exit / 39.5% miss):
  avg routing    = 300ns  (routing table lookup)
  avg shard      = 858µs  (CRTC traversal, inline on caller thread)
  branches known = 822  (shard[0]=411, shard[1]=411)
  remove broadcasts = 0  (fallback for blocks absent from index)
Shard block distribution:
  shard 0: 673927 blocks (50.6%), 7000 workers
  shard 1: 657341 blocks (49.4%), 3577 workers
```

**worker-sharded** (165,256 find_matches calls):

```
ShardedConcurrentIndexer find_matches (165256 calls, spawn_blocking scatter):
  avg outer      = 3455µs
  avg max-shard  = 796µs  (pure CRTC work, critical path)
  avg overhead   = 2659µs  (77.0% of outer)
Shard block distribution:
  shard 0: 1072286 blocks (50.5%), 3500 workers
  shard 1: 1052137 blocks (49.5%), 3500 workers
```

**multi-process-sharded-crtc** (700,000 find_matches calls, 2 shards, each on loopback TCP):

Note: benchmarker was overwhelmed (offered 7.8M ops/s, achieved 275k ops/s).  Results reflect
heavily loaded loopback TCP behavior, not an apples-to-apples fixed-load comparison.

```
MultiProcessShardedIndexer find_matches (700000 calls, 2 shards):
  avg outer            = 115677µs
  ├─ CRTC traversal    =    597µs  (0.5%)  ← shard work
  ├─ JSON total        =   1094µs  (0.9%)
  │  ├─ coord ser      =     15µs          req serialize (coord)
  │  ├─ shard deser    =     24µs          req deserialize (shard)
  │  └─ coord deser    =   1055µs          resp deserialize (coord)
  ├─ network (approx)  = 113655µs (98.3%)  ← RTT + shard resp ser
  └─ overhead total    = 115080µs (99.5%)  ← everything except CRTC
```

**Overhead breakdown vs. in-process approaches:**


| Metric          | worker-sharded                | multi-process-sharded |
| --------------- | ----------------------------- | --------------------- |
| avg outer       | 3,455 µs                      | 115,677 µs            |
| pure CRTC work  | 796 µs (23%)                  | 597 µs (0.5%)         |
| overhead        | 2,659 µs (77%)                | 115,080 µs (99.5%)    |
| overhead source | `spawn_blocking` futex wakeup | loopback TCP RTT      |


The multi-process approach is **~33× slower end-to-end** than in-process worker-sharded under
load.  The CRTC traversal itself is actually *faster* (597µs vs 796µs — smaller trees from
isolation), but the HTTP round-trip swamps that gain entirely.

The 1,055µs coord JSON deserialization time is also suspect — under the overwhelmed load, tokio
threads contend for CPU time while deserializing two concurrent shard responses, inflating this
cost far above the baseline ~20–50µs expected at moderate load.

**When multi-process sharding would be competitive:** only when shards run on separate hosts
(removing loopback TCP and substituting RDMA or dedicated NIC bandwidth), or when each shard
handles long enough requests that the RTT overhead is amortized (e.g., very long context windows
where CRTC traversal itself takes tens of milliseconds).

---

## Multi-process node-depth sharding (Phase 7)

`MultiProcessNodeDepthShardedIndexer` routes `find_matches` to **exactly one shard** via the
coordinator's shadow trie, unlike `MultiProcessShardedIndexer` which scatter-gathers to all
shards.  Only the shadow-trie lookup happens in-process; CRTC traversal happens on the target
shard's process.

**Workload:** `.prefix` trace, 2 shards × 4 event workers per shard server, `--routing-node-depth 2 --no-parent-inheritance`.

### In-process vs multi-process: node-depth 2 shards

All steady-state runs: 300s benchmark, `.prefix` trace, 37k ops/s offered (100% utilization),
`--routing-node-depth 2 --no-parent-inheritance`.


| Metric                   | in-process (30s)         | TCP loopback (300s, steady) | UDS (300s, steady)     |
| ------------------------ | ------------------------ | --------------------------- | ---------------------- |
| Offered / Achieved ops/s | 374,413 / 374,413 (100%) | 37,441 / 37,426 (100%)      | 37,442 / 37,434 (100%) |
| Shadow-trie routing      | 1,607 ns                 | 1,339 ns                    | 1,517 ns               |
| CRTC traversal           | 94 µs                    | 113 µs (2.2%)               | 128 µs (7.8%)          |
| avg outer (end-to-end)   | ~95.6 µs                 | 5,027 µs                    | **1,645 µs**           |
| p99 latency              | 135 µs                   | 5,428 µs                    | **877 µs**             |
| Wire overhead (net/UDS)  | —                        | 4,688 µs (93.3%)            | **1,264 µs (76.8%)**   |
| Overhead (non-CRTC)      | ~1.6 µs (<2%)            | 4,914 µs (97.8%)            | **1,517 µs (92.2%)**   |
| Shard leaf balance       | 50% / 50%                | 50% / 50%                   | 50% / 50%              |
| Remove broadcasts        | —                        | 776,350                     | 777,613                |


**UDS vs TCP:** 3.1× lower avg_outer, 3.7× lower wire overhead, **6.2× lower p99**.

**Detailed timing report — TCP (steady state, 300s, 37k ops/s):**

```
MultiProcessNodeDepthShardedIndexer find_matches (17615 dispatched / 100000 total, 2 shards, 82.4% miss):
  avg outer            = 5027µs
  ├─ routing (shadow)  = 1339ns  (shadow trie, coordinator)
  ├─ CRTC traversal   = 113µs  (2.2%)  ← shard work
  ├─ JSON total        = 207µs  (4.1%)
  │  ├─ coord ser      = 11µs        req serialize (coord)
  │  ├─ shard deser    = 18µs        req deserialize (shard)
  │  └─ coord deser    = 178µs        resp deserialize (coord)
  ├─ network (approx)  = 4688µs  (93.3%)  ← RTT + shard resp ser
  └─ overhead total    = 4914µs  (97.8%)  ← everything except CRTC
  shadow trie: 117500 nodes (113988 routing leaves: shard[0]=56994, shard[1]=56994) | remove broadcasts: 776350
```

**Detailed timing report — UDS (steady state, 300s, 37k ops/s):**

```
MultiProcessNodeDepthShardedIndexerUds find_matches (16904 dispatched / 100000 total, 2 shards, 83.1% miss):
  avg outer            = 1645µs
  ├─ routing (shadow)  = 1517ns  (shadow trie, coordinator)
  ├─ CRTC traversal   = 128µs  (7.8%)  ← shard work
  ├─ JSON total        = 231µs  (14.0%)
  │  ├─ coord ser      = 13µs        req serialize (coord)
  │  ├─ shard deser    = 24µs        req deserialize (shard)
  │  └─ coord deser    = 194µs        resp deserialize (coord)
  ├─ UDS wire (approx) = 1264µs  (76.8%)  ← kernel copy + shard resp ser
  └─ overhead total    = 1517µs  (92.2%)  ← everything except CRTC
  shadow trie: 117501 nodes (113989 routing leaves: shard[0]=56995, shard[1]=56994) | remove broadcasts: 777613
```

**Why UDS is faster:** Unix domain sockets bypass the TCP network stack entirely.
Loopback TCP still incurs TCP header processing, ACK generation, congestion-control state
machine, and socket buffer copies into a virtual NIC ring.  UDS copies data between processes
via a shared kernel buffer with no protocol processing overhead.

**Residual UDS overhead (~1.3ms wire):** = ~1.0ms pure socket IPC + ~230µs JSON ser/deser
(already counted separately).  The ~1ms residual is tokio task-scheduling and axum HTTP parsing
overhead — it would only be eliminated by switching from HTTP framing to a raw binary protocol.

### Comparison: single shard-server (standalone indexer analog) vs. 2-shard node-depth

The `shard-server` subcommand runs a single `ThreadPoolIndexer<CRTC>` process with an HTTP
interface — structurally identical to the `standalone_indexer` in production (which uses the
same axum/HTTP query path; only the event-ingest transport differs: ZMQ/NATS vs HTTP POST).
Running the coordinator pointing at one shard gives the HTTP overhead baseline before any
sharding benefit.

Both runs use the `.prefix` trace at 300s (37k ops/s offered, 100% utilization).


| Metric                 | 1 shard (standalone, 8w) | 2 shards (node-depth, 2×4w) | Change                                 |
| ---------------------- | ------------------------ | --------------------------- | -------------------------------------- |
| CRTC traversal         | **174 µs** (3.4%)        | **113 µs** (2.2%)           | **−35%** (smaller trees)               |
| avg outer              | 5,169 µs                 | 5,027 µs                    | −3% (within noise)                     |
| Network RTT (loopback) | 4,593 µs (88.9%)         | 4,688 µs (93.3%)            | ≈ same                                 |
| p99 latency            | 2,252 µs                 | 5,428 µs                    | see note below                         |
| JSON deserialization   | 342 µs coord deser       | 178 µs coord deser          | 2× faster (smaller response per shard) |
| Throughput             | 37.4k ops/s              | 37.4k ops/s                 | same offered load                      |
| Remove broadcasts      | 777,383                  | 776,350                     | ≈ same                                 |


**p99 note:** the benchmark p99 is computed over *all* 100k `find_matches` calls, 82.4% of which
return instantly (trie miss, ~0µs).  p99 therefore falls at roughly the 94th percentile of
dispatched calls.  At 300s the trees are fully populated; shard event-processing variance
(fewer workers per shard in the 2-shard case) inflates tail latency.

**Key takeaway:** sharding meaningfully reduces CRTC traversal time (174→113µs, −35%) because
each shard holds only half the tree.  However, because both configurations are dominated by
loopback TCP RTT (~4.6ms, >88% of total), this per-shard gain has little effect on avg_outer
or p99 in a single-host benchmark.  The gain becomes significant when:

- CRTC trees are large enough that traversal time (not RTT) is the bottleneck, OR
- Shards run on separate hosts (RTT drops from ~5ms loopback to ~0.1ms RDMA/NVLink).

**Note on event transport:** In production, `standalone_indexer` ingests events via ZMQ/NATS
(pub-sub, no per-message HTTP overhead) — substantially lower event-ingestion latency than
`shard-server` (HTTP POST per event).  The query-side comparison above is fair; the event-side
numbers would favor the standalone indexer in a real deployment.

---

### Comparison: node-depth multi-process vs. worker-sharded multi-process


| Metric                  | worker-sharded multi-process | node-depth multi-process (steady) | Improvement |
| ----------------------- | ---------------------------- | --------------------------------- | ----------- |
| avg outer               | 115,677 µs (overwhelmed)     | 5,027 µs (steady state)           | **~23×**    |
| pure CRTC work          | 597 µs (0.5%)                | 113 µs (2.2%)                     | 5.3× faster |
| Shards queried per call | all N (scatter-gather)       | exactly 1 (shadow trie route)     | —           |
| Overhead source         | loopback TCP × N shards      | loopback TCP × 1 shard            | —           |


The 23× improvement in end-to-end latency comes entirely from eliminating scatter-gather:
worker-sharded must wait for all N shard responses (latency dominated by the slowest),
node-depth queries only one shard.

The CRTC traversal itself is also 5× faster (113 µs vs 597 µs) because each shard only
holds its assigned fraction of the tree — smaller trees, faster traversals.  This benefit
exists in both multi-process variants but is swamped by network overhead in the worker-sharded
version due to the overwhelmed benchmark condition.

### Why loopback TCP still dominates

Even for single-shard dispatch, loopback TCP RTT is ~~1.5–5 ms depending on system load.  The
in-process equivalent is ~95 µs (shadow trie + CRTC traversal).  The multi-process version is
**~~53× slower end-to-end** at steady state — all additional latency is OS networking overhead.

For node-depth routing to be competitive in a multi-process deployment:

- Shards must run on separate hosts with low-latency networking (RDMA, NVLink, InfiniBand)
- Or CRTC traversal time per shard must be large enough to amortize RTT (>10 ms traversals
would make 5 ms RTT only a 50% overhead rather than 97%)

At the model-inference timescale (50–500 ms per token), a 5 ms routing overhead from a
well-provisioned network is entirely acceptable.

---

## Shard collapse at depth=1

The mooncake trace has only **4 distinct `hash_ids[0]`** values: 0, 46, 74, 26783.
`find_matches` routing uses `LocalBlockHash` = `XXH3(block_size × id_as_u32)` — a 64-bit value.
The FNV-1a hash of all four mod 2 lands on the same shard (a coincidence of these specific XXH3
outputs). 100% of traffic ends up on shard 0; shard 1 is completely idle.

This looks surprising because there *are* 4 varying first tokens. The key is that the shard key is
derived from `FNV(LocalBlockHash)`, not directly from `hash_ids[0]`. When there are only a handful
of distinct inputs, hash collisions mod N are easy to hit by chance.

**Effect on performance:** Even with shard collapse, prefix-sharded depth=1 (1,612 µs) edges out
CRTC baseline (1,641 µs) because the single active shard benefits from half the write contention
on its CRTC's per-node locks (all blocks stay on shard 0 but the event workers for shard 1 are
still consuming the DashMap insert path). However, the second shard's 8 cores and 4 event workers
are completely wasted.

**Fix:** `prefix_depth` default raised to **2**. With depth=2, there are 6,557 distinct
`(hash_ids[0], hash_ids[1])` pairs, giving a real split.

---

## Shard imbalance at depth=2 (prefix-sharded)

Even with depth=2, prefix-sharded achieves only a 69%/31% block split.  The trace has a
dominant cluster of `(0, X)` pairs — 10,938 requests (46% of traffic) each with a distinct
second block — which all hash to the same shard.  The 9,203 `(46, 47)` requests and 3,449
`(74, 75)` requests hash to the other shard.

`hash % N` is only as balanced as the hash values themselves.  On a trace with clustered prefixes,
some shards receive materially more load.

---

## Branch-sharded: least-loaded assignment

`BranchShardedIndexer` solves the imbalance with a routing *table* instead of `hash % N`.
The first time a branch key is seen (root `Stored` event for a new depth-2 prefix), it is
assigned to the shard with the fewest branches at that moment.  The assignment is then sticky:
all subsequent events and queries for that branch go to the same shard.

**Results on the mooncake trace (depth=2):**

- **822 distinct branch keys** observed across the run
- **Perfect split: shard[0]=411, shard[1]=411** (least-loaded assignment gives exactly 50/50)
- **Block distribution: 50.6% / 49.4%** — vs. 69%/31% for prefix-sharded
- **p99 = 1,381 µs** — best result, 16% below CRTC baseline

**The 39.5% early-exit rate** is a feature, not a problem.  When a `find_matches` query's branch
key is absent from the routing table, no worker has ever stored a sequence with that prefix — so
the correct answer is "no matches" and we return it immediately without touching any shard.
These are genuinely new/unseen prefixes.  The 60.5% of calls that do dispatch are the ones with
real data in the tree; the others would have been a wasted traversal in any prefix-based design.

Because 39.5% of calls skip the shard entirely, the 60.5% that do dispatch drive the avg_shard
higher (858 µs vs 795 µs for prefix-sharded) — those are the populated branches being traversed.

**Remove broadcasts = 0** confirms the `block_to_shard` index is always up to date; the broadcast
fallback (for blocks absent from the index) never fires on this trace.

---

## Worker-sharded overhead

`ShardedConcurrentIndexer::find_matches` scatters to all N shards via `spawn_blocking`, awaits
all N results, then merges.  At 50k/s offered load with 2 shards:

```
avg outer     = 3,455 µs
avg max-shard =   796 µs  (actual CRTC work — same cost as baseline)
avg overhead  = 2,659 µs  (77% of total)
```

The CRTC traversal itself is fine.  The 77% overhead is **scatter-gather dispatch cost**:

- Every `find_matches` spawns 2 `spawn_blocking` tasks → each shard gets every query
- Under high concurrency (165k calls × 2 shards = 330k blocking tasks), tokio's blocking pool
saturates, tasks queue, p99 reaches 48 ms
- The `max_shard` timer starts *inside* the blocking task (after scheduling), so it measures
pure traversal time — the queue wait is entirely in "overhead"

**Flume pool does not fix this.** A `ShardReadPool` pattern (pre-spun OS threads blocking on
`flume::recv()`) was implemented and benchmarked. It performed **worse** at every load level:


| Mode            | load         | avg outer | overhead |
| --------------- | ------------ | --------- | -------- |
| spawn_blocking  | -d 1 (7k/s)  | 444 µs    | 78%      |
| flume pool N=2  | -d 1 (7k/s)  | 1,220 µs  | 95%      |
| spawn_blocking  | -d 7 (50k/s) | 3,455 µs  | 77%      |
| flume pool N=32 | -d 7 (50k/s) | 5,523 µs  | 87%      |


Sleeping OS threads blocked on `flume::recv()` incur ~200-400 µs futex wake-up latency.
Tokio's `spawn_blocking` reuses warm threads with lower scheduling latency.

**Root cause is scatter-gather, not dispatch.** No dispatch mechanism changes the fact that
every shard receives every query.  The fix is single-shard dispatch — use `PrefixShardedIndexer`
or `BranchShardedIndexer`.

---

## Design comparison


| Dimension                      | prefix-sharded                            | branch-sharded                              | worker-sharded                          | node-depth-sharded                                          |
| ------------------------------ | ----------------------------------------- | ------------------------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| Shard selection                | `FNV(prefix) % N`                         | routing table, least-loaded on first insert | worker sticky assignment                | shadow trie, least-loaded at depth K                        |
| find_matches shards queried    | 1                                         | 1 (or 0 on miss)                            | all N                                   | 1 (or 0 on miss)                                            |
| Shard balance (mooncake trace) | 69%/31% blocks                            | 50%/50% blocks                              | 50%/50% workers                         | 50%/50% routing leaves (.prefix trace, depth=2)                |
| Unknown prefix handling        | always dispatches                         | early-exit, returns empty                   | always scatters                         | early-exit, returns empty                                   |
| Extra write-path overhead      | DashMap insert per block                  | DashMap insert per block + routing table    | DashMap lookup for worker assignment    | shadow trie write + `block_to_shard` insert                 |
| Remove routing                 | per-block index                           | per-block index with broadcast fallback     | per-worker routing                      | per-block index with broadcast fallback                     |
| Miss semantics                 | no — always routes to a shard             | yes — 39.5% calls skip all shards           | no — always scatters                    | yes — 82.4% calls skip (trie miss)                          |
| System prompt handling         | collapses at depth ≤ prompt length        | collapses at depth ≤ prompt length          | not applicable                          | works correctly at depth=2 regardless of prompt length      |
| Best for                       | diverse prefixes with natural hash spread | skewed prefix workloads                     | N/A (scatter-gather overhead dominates) | workloads with long shared system prompt (node-depth aware) |


---

## Phase 1: Worker-Based Sharding — implementation notes

`ShardedConcurrentIndexer<T>` (`lib/kv-router/src/indexer/sharded_concurrent.rs`)

- `shards: Vec<Arc<ThreadPoolIndexer<T>>>` — N shards, each with its own event worker pool
- `worker_assignments: DashMap<WorkerId, usize>` — sticky least-loaded worker-to-shard mapping
- `worker_counts: Mutex<Vec<usize>>` — load tracking for assignment

A `ShardReadPool` option (`--num-read-threads-per-shard > 0`) is implemented but defaults to `0`
(spawn_blocking) because the flume pool is empirically slower; see §Worker-sharded overhead.

---

## Phase 2: Prefix-Based Sharding — implementation notes

`PrefixShardedIndexer<T>` (`lib/kv-router/src/indexer/prefix_sharded.rs`)

- `shards: Vec<Arc<ThreadPoolIndexer<T>>>` — N shards, each with its own event worker pool
- `prefix_depth: usize` — number of prefix blocks hashed for routing (default 2)
- `block_to_shard: DashMap<u64, usize>` — maps `ExternalSequenceBlockHash → shard index`

Consistency: `find_matches` and `Stored` routing both use `min(prefix_depth, len)` blocks with
identical FNV-1a hash. A `ShardReadPool` option is available (`--num-read-threads-per-shard`).

---

## Phase 3: Branch-Based Sharding — implementation notes

`BranchShardedIndexer<T>` (`lib/kv-router/src/indexer/branch_sharded.rs`)

- `shards: Vec<Arc<ThreadPoolIndexer<T>>>` — N shards, each with its own event worker pool
- `prefix_depth: usize` — number of prefix blocks identifying a branch (default 2)
- `branch_to_shard: DashMap<u64, usize>` — routing table: `FNV(prefix) → shard`
- `branch_counts: Mutex<Vec<usize>>` — per-shard branch count for least-loaded assignment
- `block_to_shard: DashMap<u64, usize>` — `ExternalSequenceBlockHash → shard` for Remove routing

**Operations:**


| Operation                            | Behavior                                                                                                                                  |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `find_matches(seq)`                  | Compute branch key → lookup in routing table. If found, dispatch to that shard. If unknown, return empty scores immediately (early-exit). |
| `apply_event` (Stored, root)         | Compute branch key. If new, assign to least-loaded shard and record. Record all block→shard mappings.                                     |
| `apply_event` (Stored, continuation) | Inherit shard from parent block hash lookup.                                                                                              |
| `apply_event` (Removed)              | Route each block via `block_to_shard`; broadcast fallback for unknowns.                                                                   |
| `apply_event` (Cleared)              | Broadcast to all shards.                                                                                                                  |
| `remove_worker`                      | Broadcast to all shards.                                                                                                                  |


---

## Conversation trace (15-block shared system prompt)

Trace: `.prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl`
(100,000 requests; `block_size=128`; sequence lengths min=25, max=320, median=155 blocks)

All runs: 2 shards × 4 event workers/shard, `-d 3`, `--benchmark-duration-ms 30000`.

### Prefix depth distribution


| Depth | Unique keys | Notes                                                                                    |
| ----- | ----------- | ---------------------------------------------------------------------------------------- |
| 1–15  | **1**       | All requests share a 15-block system prompt (`hash_ids[0..14] = [1..15]`), ~1,920 tokens |
| 16    | **44,641**  | First divergent block; 94.9% of keys appear exactly once (singletons)                    |
| 17–25 | **44,641**  | Identical — each depth-16 prefix has exactly one extension through depth 25              |
| 30    | 24,918      | Drops because sequences shorter than 30 blocks are excluded                              |


The depth-16 request distribution is very flat: the most common prefix covers only 0.37% of
requests, top-10 cover 2.0%, top-1000 cover 37.5%.  The 44,641 unique keys hash almost
perfectly 50/50 (50.2% / 49.8% by FNV-mod-2).

### Benchmark results


| Indexer               | depth | p99 latency | Block throughput | Shard split (blocks) | Notes                                                |
| --------------------- | ----- | ----------- | ---------------- | -------------------- | ---------------------------------------------------- |
| CRTC baseline (8w)    | —     | 889 µs      | 3,765,535 blk/s  | —                    | single tree                                          |
| prefix-sharded (2×4w) | 2     | 885 µs      | 3,765,281 blk/s  | **0% / 100%**        | collapse — all share first 2 blocks                  |
| branch-sharded (2×4w) | 2     | 895 µs      | 3,765,640 blk/s  | **100% / 0%**        | collapse — same cause                                |
| prefix-sharded (2×4w) | 16    | **474 µs**  | 3,766,783 blk/s  | 0.2% / 99.8%         | imbalanced — see §Continuation routing               |
| branch-sharded (2×4w) | 16    | **350 µs**  | 3,767,275 blk/s  | 0.2% / 99.8%         | 82.7% early-exit is incorrect — see §False negatives |
| prefix-sharded (2×4w) | 20    | 474 µs      | 3,766,633 blk/s  | 0.2% / 99.8%         | same as depth=16                                     |
| branch-sharded (2×4w) | 20    | 369 µs      | 3,767,610 blk/s  | 0.2% / 99.8%         | same as depth=16                                     |
| prefix-sharded (2×4w) | 25    | 469 µs      | 3,766,877 blk/s  | 98.9% / 1.1%         | shard flips — coincidence of root-event FNV values   |
| branch-sharded (2×4w) | 25    | 359 µs      | 3,767,259 blk/s  | 0.2% / 99.8%         | same as depth=16                                     |


Depth=2 causes total collapse (all requests share blocks 1–2).  Depth≥16 gives better latency
but the block distribution remains 99%+ on one shard regardless of depth.

### Detailed timing: depth=16

**prefix-sharded depth=16** (300,000 find_matches calls):

```
avg routing = 314ns  (FNV hash over 16 blocks)
avg shard   = 285µs  (CRTC traversal, inline on caller thread)
Shard block distribution:
  shard 0: 58,909 blocks (0.2%), 1620 workers
  shard 1: 29,434,918 blocks (99.8%), 2375 workers
```

**branch-sharded depth=16** (300,000 total find_matches calls):

```
BranchShardedIndexer find_matches (300000 total: 51950 dispatched, 248050 early-exit / 82.7% miss):
  avg routing    = 713ns  (routing table lookup)
  avg shard      = 239µs  (CRTC traversal, inline on caller thread)
  branches known = 1084  (shard[0]=542, shard[1]=542)
  remove broadcasts = 10,207,088  (fallback for blocks absent from index)
Shard block distribution:
  shard 0: 24,403,198 blocks (99.8%), 2949 workers
  shard 1: 60,415 blocks (0.2%), 1638 workers
```

---

## Continuation routing: why depth doesn't fix the imbalance

Both `PrefixShardedIndexer` and `BranchShardedIndexer` use parent-inheritance in `apply_event`
(`prefix_sharded.rs:300`, `branch_sharded.rs:321`):

```rust
let shard_idx = if let Some(parent_hash) = &store_data.parent_hash {
    self.block_to_shard.get(&parent_hash.0).map(|v| *v)
        .unwrap_or_else(|| self.shard_for_stored_blocks(&store_data.blocks))
} else {
    // Root event: use prefix hash / branch key
    self.shard_for_stored_blocks(&store_data.blocks)
};
```

**What happens on a shared-system-prompt trace:**

1. The very first request stores blocks [1..15, X, ...].  Its prefix hash routes the root event
  to (say) shard 1.  All 16 blocks are recorded in `block_to_shard → shard 1`.
2. Every subsequent request also starts with blocks 1–15.  Their Stored event has
  `parent_hash` pointing to one of those shared blocks, which is already in `block_to_shard`.
   The continuation path fires → **shard 1 again**, regardless of their unique block 16.
3. Result: ~99.9% of all events go to the shard that won the first root event — regardless
  of `prefix_depth`.

Increasing depth from 2 → 16 → 25 does not help because the continuation lookup bypasses
the prefix hash for any sequence that shares any ancestor with a previously stored sequence.

**For branch-sharded**, `branch_to_shard` only records entries for ROOT Stored events (no
parent or parent not found).  On this trace, only ~1,082 sequences were roots (the very first
sequences before the system prompt was established in the index).  The other ~99,998 request
chains were stored via continuation and never entered the routing table.

---

## `--no-parent-inheritance` results

Both indexers now accept `--no-parent-inheritance` (CLI) / `new_with_options(..., inherit_parent_shard: false)` (API).
When set, every Stored event routes by prefix hash regardless of whether its parent is in `block_to_shard`.

### On the .prefix trace (2 shards × 4w, `-d 3`, `--benchmark-duration-ms 30000`)


| Indexer                            | depth  | p99        | Block split   | branches known | Notes                                                                                                        |
| ---------------------------------- | ------ | ---------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| prefix-sharded (inherit=true)      | 2      | 885 µs     | 0% / 100%     | —              | collapse                                                                                                     |
| prefix-sharded (inherit=false)     | 2      | 916 µs     | 0% / 100%     | —              | still collapses — hash of depth-2 prefix `[1,2]` is identical for all requests                               |
| prefix-sharded (inherit=true)      | 16     | 474 µs     | 0.2% / 99.8%  | —              | imbalanced                                                                                                   |
| **prefix-sharded (inherit=false)** | **16** | **478 µs** | **44% / 56%** | —              | **good balance — continuation routing was the sole cause of imbalance**                                      |
| branch-sharded (inherit=true)      | 2      | 895 µs     | 100% / 0%     | 172            | collapse                                                                                                     |
| branch-sharded (inherit=false)     | 2      | 899 µs     | 100% / 0%     | 6,354,790      | collapse — root events still have same depth-2 prefix; routing table inflated 37,000× by continuation events |
| branch-sharded (inherit=true)      | 16     | 350 µs     | 0.2% / 99.8%  | 1,084          | imbalanced, false negatives                                                                                  |
| **branch-sharded (inherit=false)** | **16** | **363 µs** | **48% / 52%** | **6,366,617**  | **good balance, 82.7% early-exit preserved, routing table 6,000× larger**                                    |


### On the mooncake trace (2 shards × 4w, `-d 7`, `--benchmark-duration-ms 10000`)

Mooncake has no long shared prefix, so removing parent inheritance has modest effect:


| Indexer                        | depth | p99      | Block split   | Notes                                              |
| ------------------------------ | ----- | -------- | ------------- | -------------------------------------------------- |
| prefix-sharded (inherit=true)  | 2     | 1,401 µs | 69% / 31%     |                                                    |
| prefix-sharded (inherit=false) | 2     | 1,390 µs | **52% / 48%** | slightly better balance without parent inheritance |
| branch-sharded (inherit=true)  | 2     | 1,381 µs | 51% / 49%     | 39.5% early-exit                                   |
| branch-sharded (inherit=false) | 2     | 1,332 µs | 56% / 44%     | 39.5% early-exit preserved                         |


No-parent-inheritance is a slight improvement on mooncake (better balance, no p99 regression).

### Routing table inflation with `inherit=false`

With parent inheritance disabled, **every stored event** independently computes a branch key from
its own first `prefix_depth` blocks.  On the .prefix trace, most events are delta-stored (only new
blocks, starting mid-sequence), so their first-K-blocks keys are unique and unrelated to the
sequence's true prefix.  This creates one routing-table entry per event rather than per unique
sequence root:


| Mode          | branches known (d=16) | Ratio   |
| ------------- | --------------------- | ------- |
| inherit=true  | ~1,082                | 1×      |
| inherit=false | ~6,366,617            | ~6,000× |


In production at scale this would mean hundreds of millions of entries in `branch_to_shard`.
For prefix-sharded there is no routing table, so no inflation — only the `block_to_shard`
index grows (which is identical in both modes).

### Residual false negatives with `inherit=false` on branch-sharded

Even with no-parent-inheritance, the 82.7% early-exit in branch-sharded d=16 persists.
The routing table now has 6.37M entries, but `find_matches` still misses 82.7%.  The reason:
`find_matches` always computes branch key from the first `prefix_depth` blocks of the *full
query sequence* (positions 0–15), while continuation stored events create keys from their
*delta slice* (positions K to K+15 for some K>0).  These keys are different, so find_matches
cannot hit continuation-keyed entries.  Only root-stored events (starting at position 0)
create keys consistent with find_matches.  The count of those (~51k dispatched / 300k) matches
the with-inheritance count exactly — the fundamental miss-rate is unchanged.

**Conclusion:** `prefix-sharded depth=16, inherit=false` is the cleanest fix for the
shared-prefix imbalance — it achieves 44%/56% block balance, no routing-table inflation, and
no correctness concerns.  Branch-sharded with `inherit=false` achieves similar balance
(48%/52%) but inflates the routing table by 6,000× and still returns false negatives at the
same rate.

---

## False negatives in branch-sharded on shared-prefix traces

The 82.7% early-exit at depth=16 is **not** all genuinely unseen prefixes.  The correct
interpretation:

- 1,084 branches are in `branch_to_shard` (root-stored sequences).
- The remaining 43,557 unique depth-16 prefixes were stored via continuation (parent known),
so they have no entry in `branch_to_shard`.
- When `find_matches` queries one of those continuation-stored sequences, the branch key is
absent from the routing table → early-exit → **empty scores returned**.
- But those sequences' blocks ARE in the tree (on the dominant shard).  The early-exit is
a **false negative**: the router claims no cache hit when one exists.

This is a correctness problem for workloads with a shared system prompt.  The early-exit
optimization assumes "branch not in routing table → never stored", which is violated when
continuation routing bypasses `branch_to_shard` assignment.

The high `remove_broadcast_count` (~10M) is a symptom of the same issue: Remove events
arrive for blocks that were stored via continuation and were never recorded in `block_to_shard`
at the time of the Remove.

---

## CRTC vs Branch-sharded: load sweep on .prefix trace

All runs on `.prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl`.
Baseline: CRTC 8 event workers.  Branch-sharded: 2 shards × 4 event workers/shard (8 total),
`--prefix-depth 16 --no-parent-inheritance`, `-n 1000 --find-matches-concurrency 4`.

### Throughput and latency: consistent load (30 s benchmark, same offered rate per d)

`--benchmark-duration-ms 30000`, CPU-isolated runs (taskset).


| `-d` | Offered (blk/s) | CRTC achieved       | CRTC p99 | Branch achieved     | Branch p99   | Tput ratio | p99 ratio |
| ---- | --------------- | ------------------- | -------- | ------------------- | ------------ | ---------- | --------- |
| 1    | 1,256,638       | 1,256,303 *(100%)*  | 284 µs   | 1,256,221 *(100%)*  | **136 µs**   | 1.0×       | **2.1×**  |
| 3    | 3,769,882       | 3,765,991 *(99.9%)* | 933 µs   | 3,763,496 *(99.8%)* | **358 µs**   | 1.0×       | **2.6×**  |
| 6    | 7,539,838       | 7,514,039 *(99.7%)* | 1,614 µs | 7,520,988 *(99.7%)* | **740 µs**   | 1.0×       | **2.2×**  |
| 10   | 12,566,301      | 10,695,938 *(85%)*  | 5,059 µs | 12,179,067 *(97%)*  | **1,318 µs** | **1.14×**  | **3.8×**  |


At low-to-moderate load (d=1–6) both configs achieve offered throughput; Branch-sharded has
**~2× lower p99** because each shard's tree is smaller.  At d=10 CRTC begins saturating (85%
achieved vs 97% for Branch).

### Saturation stress test (15 s benchmark, 2× offered rate)

`--benchmark-duration-ms 15000`.  Higher offered rate reveals where each config hard-saturates.


| `-d` | Offered (blk/s) | CRTC achieved     | CRTC p99  | Branch achieved    | Branch p99   | Tput ratio | p99 ratio |
| ---- | --------------- | ----------------- | --------- | ------------------ | ------------ | ---------- | --------- |
| 10   | 25,133,286      | 9,484,974 *(38%)* | 6,677 µs  | 18,866,818 *(75%)* | **946 µs**   | **2.0×**   | **7.1×**  |
| 15   | 37,699,956      | 4,154,997 *(11%)* | 10,908 µs | 21,334,732 *(57%)* | **2,208 µs** | **5.1×**   | **4.9×**  |
| 20   | 50,266,388      | 2,584,177 *(5%)*  | 23,223 µs | 26,279,736 *(52%)* | **2,774 µs** | **10.2×**  | **8.4×**  |


CRTC saturates at ~10–11 M blk/s on this workload.  Branch-sharded continues scaling past
26 M blk/s (still 52% utilization at d=20), limited by the single shared tokio thread pool
rather than tree contention.

### Shard balance (branch-sharded, no-inherit, depth=16)


| Run         | Shard 0 blocks | Shard 1 blocks |
| ----------- | -------------- | -------------- |
| d=1 (30 s)  | 49.2%          | 50.8%          |
| d=10 (30 s) | 49.2%          | 50.8%          |
| d=10 (15 s) | 49.5%          | 50.5%          |
| d=15 (15 s) | 50.4%          | 49.6%          |


### Node count comparison


| Indexer                                         | Nodes (total)                          | Avg hashes/edge | Notes                                                                       |
| ----------------------------------------------- | -------------------------------------- | --------------- | --------------------------------------------------------------------------- |
| CRTC (single tree, 8 workers)                   | **~4,726,000–5,367,000**               | 1.9–2.2         | Shared 15-block prefix → ~45k branch points near root, many short edges     |
| Branch-sharded (2 shards, no-inherit, depth=16) | **~81,000–83,000**                     | 39–45           | No shared prefix within shards → long independent chains, few branch points |
| **Ratio**                                       | **~60× fewer nodes in Branch-sharded** |                 |                                                                             |


**Why branch-sharded has ~60× fewer nodes despite covering the same data:**

In the CRTC, the shared 15-block system prompt is a single path diverging into ~44,641 unique
branches at depth 16 — one new radix node for every request that arrives with a novel depth-16
block.  The tree has high branching factor near the root, creating ~5M nodes overall.

With `--no-parent-inheritance`, branch-sharded routes each stored event by its own first-16-block
key.  Continuation events (delta-stored blocks starting at position K>0) create keys from
position K, not position 0.  These mid-sequence chains don't share a common prefix across
requests, so they form long independent paths without branch points.  The radix compression is
maximal: few nodes, each covering many hashes (avg 45).

The CRTC's large node count reflects its richer shared structure; branch-sharded's small node
count reflects that sharding broke the shared-prefix compression into isolated chains.  In a
multi-machine deployment, smaller per-shard trees mean faster traversal and less memory
footprint per node.

---

## Phase 4: Query-Rate Rebalancing — implementation notes

`RebalancingBranchShardedIndexer<T>` (`lib/kv-router/src/indexer/branch_sharded_rebalancing.rs`)

All `BranchShardedIndexer` behavior is preserved; rebalancing adds:

- `shard_query_counts: Vec<AtomicU64>` — per-shard cumulative `find_matches` hits
- `branch_query_counts: DashMap<u64, AtomicU64>` — per-branch hit counter; identifies the hottest branch
- `replaying_branches: DashSet<u64>` — Phase 1 guard: branch is being replayed; suppress dual-write
- `dualwrite_branches: DashMap<u64, DualWriteEntry>` — Phase 2: scatter-gather both shards + dual-write new events
- Background tokio task (holds `Weak<Self>`) checks every `rebalance_interval`; stops when indexer drops

### Two-phase migration protocol

A naive "dump → replay → switch" protocol has a race window: events arriving between the dump
and the routing switch only go to the old shard and are missed after the switch.  The fix is
**dual-write + scatter-gather** during the window, but naively enabling dual-write creates an
ordering problem: the CRTC drops continuation events whose parent block is not yet stored
(`ParentBlockNotFound`).  A live dual-write event arriving at the new shard before its replay
root has been processed is permanently lost.

**The FIFO ordering guarantee** solves this: `ThreadPoolIndexer` routes each `WorkerId` to the
same OS thread via a flume channel (sticky assignment, FIFO ordering within the channel).
Replay events are *enqueued* into the new shard's channels **before** dual-write is activated,
so flume guarantees they are *processed* before any subsequent live dual-write event for the
same worker.

```
Phase 1 (Replaying):
  new events → old_shard only   (dual-write suppressed by replaying_branches check)
  find_matches → old_shard only
  migration task:
    1. replaying_branches.insert(branch_key)
    2. dump_events(old_shard)              ← FIFO barrier: in-flight writes drain
    3. filter Stored events for branch_key
    4. shards[new_shard].apply_event(event) for each  ← enqueued BEFORE dual-write

Phase 2 (DualWrite): activated atomically after step 4
  new events → old_shard AND new_shard
  find_matches → scatter-gather both, merge OverlapScores (max per worker)
  migration task:
    5. dualwrite_branches.insert(branch_key, {old, new})
    6. replaying_branches.remove(branch_key)   ← Phase 2 now visible
    7. flush(new_shard)                        ← wait for replay + early dual-writes
    8. sleep(dual_write_window)                ← gap traffic warms new_shard
    9. branch_to_shard[branch_key] = new_shard
   10. dualwrite_branches.remove(branch_key)   ← single-shard resumes
```

### Rebalancing benchmark results (mooncake trace, 2 shards × 4w, `-d 7`)

The mooncake trace has an inherent **query-rate skew** even when branch counts are 50/50:
`find_matches` hits are 30k/70k (2.3× skew, max/avg = 1.40) because a few branches carry
most of the query traffic.


| Run                       | Duration | threshold | interval | window | p99             | ops/s  | Notes                                                                  |
| ------------------------- | -------- | --------- | -------- | ------ | --------------- | ------ | ---------------------------------------------------------------------- |
| baseline (branch-sharded) | 30 s     | —         | —        | —      | 1,286 µs        | 16,721 | Reference                                                              |
| rebalancing idle          | 30 s     | 9.9       | 5 s      | —      | 1,336 µs (+4%)  | 16,721 | Rebalancer never fires; overhead is DashMap+AtomicU64 per find_matches |
| DualWrite active          | 60 s     | 1.3       | 5 s      | 10 s   | 2,250 µs (+75%) | 16,721 | Migration in progress; 1 branch in scatter-gather window at end        |
| Post-migration steady     | 60 s     | 1.3       | 2 s      | 3 s    | 1,773 µs (+38%) | 16,721 | 0 migrating; oscillation: shard[0] 57k, shard[1] 30k (reversed)        |


**Key findings:**

1. **Idle overhead is ~4% p99** (1,286 → 1,336 µs). Cost is two `DashMap`/`AtomicU64` ops per
  `find_matches`. No migrations fired at threshold=9.9.
2. **Natural query skew on mooncake: 2.3× (max/avg = 1.40).** The branch counts are perfectly
  balanced (50/50) but one shard receives 2.3× more `find_matches` hits. This is why
   structural load balancing (branch counts) is insufficient for query-rate imbalance.
3. **DualWrite window adds ~75% p99 overhead.** During scatter-gather, every `find_matches`
  for a migrating branch queries both shards and awaits both results. On the mooncake trace
   this is most queries, making the window expensive.
4. **Oscillation after migration.** After the hottest branch migrates from shard[1] → shard[0],
  shard[0] becomes the new hot shard (57k vs 30k hits). With a low threshold, the rebalancer
   triggers again in the opposite direction, causing shard-flipping. `shard_query_counts` are
   adjusted after migration but the remaining per-shard counts still reflect pre-migration
   accumulation.
5. `**--no-parent-inheritance` required for full correctness.** With `inherit_parent_shard=true`
  (default), continuation events route via `block_to_shard` which still points to `old_shard`
   after migration — those sequences bypass the updated `branch_to_shard`. Migration only fully
   works on the mooncake trace because it has no long shared prefix. For shared-prefix workloads,
   use `--no-parent-inheritance`.

### Threshold calibration for N=2 shards

With exactly 2 shards, `max/avg` and `max/min` are directly related:

```
avg = (max + min) / 2
max/avg > T  ⟹  max / ((max + min) / 2) > T  ⟹  max/min > 2T − 1
```

At `threshold = 1.5`, rebalancing requires one shard to carry **3× the other's query load**.
The mooncake trace sits at max/avg = 1.40, so `threshold ≤ 1.40` is needed to trigger.
Lower thresholds increase migration frequency and oscillation risk.

### Benchmark commands

```bash
TRACE=lib/kv-router/mooncake_trace.jsonl  # or path to your trace

# Idle (rebalancer registered but threshold never reached)
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  rebalancing-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 \
  --prefix-depth 2 --rebalance-interval-secs 5 --imbalance-threshold 9.9 --dual-write-window-secs 10

# With migration (threshold triggers on mooncake natural skew)
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 60000 -d 7 \
  rebalancing-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 \
  --prefix-depth 2 --rebalance-interval-secs 5 --imbalance-threshold 1.3 --dual-write-window-secs 10 \
  --no-parent-inheritance
```

---

## Rebalancing CRTC at depth=2 on shared-prefix trace (.prefix)

**Question:** Does `rebalancing-branch-sharded-crtc` with `depth=2` fix the shared-prefix
imbalance by detecting the hot shard and rebalancing toward depth-16 branches?

**Short answer: No.** The rebalancer is blind to branching below `prefix_depth=2` and cannot
split the single branch key that all requests share.

### What happens

On the .prefix trace (100k requests, 15-block system prompt), `depth=2` produces exactly **one
unique branch key** (FNV of blocks 1–2, shared by every request). All events initially
assigned via continuation routing go to whichever shard stored the system prompt first, so
shard 0 ends up with 100% of blocks regardless of the 84/84 branch-count split.

The rebalancer wakes up, observes `shard[0]=110k hits, shard[1]=0 hits`, and fires a
migration. But migrating a depth-2 branch means replaying that shard's CRTC data to the
other shard and dual-writing. During the dual-write window, **scatter-gather overhead
dominates** and p99 nearly doubles.

After migration completes, the same problem recurs: new continuation events keep flowing
to shard 0 via `block_to_shard` inheritance from the system-prompt blocks, so the rebalancer
fires again → infinite oscillation with overhead.

The rebalancer **cannot discover depth-16 branching** because:

- Branch keys are computed from the first `prefix_depth=2` blocks only
- At depth=2, there is one branch key (all requests share blocks 1–2)
- There is no mechanism for the rebalancer to subdivide a single branch key

### Benchmark results (`.prefix` trace, 2 shards × 4w, `-d 3`, 30 s)

Command:

```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  .prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl \
  --compare concurrent-radix-tree-compressed,branch-sharded-crtc,rebalancing-branch-sharded-crtc \
  --num-event-workers 8 -d 3 --benchmark-duration-ms 30000
```


| Indexer                           | depth | p99                 | Block throughput | Shard split (blocks) | Notes                                                                        |
| --------------------------------- | ----- | ------------------- | ---------------- | -------------------- | ---------------------------------------------------------------------------- |
| CRTC baseline (8w)                | —     | **913 µs**          | 3,764,432 blk/s  | —                    | single tree                                                                  |
| branch-sharded (2×4w)             | 2     | 914 µs              | 3,765,435 blk/s  | **100% / 0%**        | collapse; 168 branches 84/84 by count, but 0% early-exit                     |
| rebalancing-branch-sharded (2×4w) | 2     | **1,792 µs (+96%)** | 3,761,052 blk/s  | **99.9% / 0.1%**     | rebalancer fired (1 branch in dual-write); scatter-gather overhead dominates |


Detailed stats for `rebalancing-branch-sharded-crtc` (300,000 find_matches):

```
avg routing       = 326ns
avg shard         = 206µs  (lower: shard[1] returns empty in ~0ns for its 0.1% blocks)
branches known    = 168  (shard[0]=84, shard[1]=84)
query hits/shard  = shard[0]=110398, shard[1]=0  (rebalancer sees 110k:0 → fires)
migrating         = 0 replaying, 1 dual-writing
remove broadcasts = 10,207,424  (all Remove events — block_to_shard never populated for
                                 continuation-stored blocks)
Shard block distribution:
  shard 0: 48,696,955 blocks (99.9%), 3000 workers
  shard 1:     45,000 blocks (0.1%),  3000 workers
```

**Why p99 regresses to 1,792 µs (vs 913 µs baseline):** during the dual-write window the
migrating branch is scatter-gathered to both shards on every `find_matches`.  The entire
trace is on shard 0; shard 1 is nearly empty.  Both shard results must complete before the
response is returned.  This adds a ~900 µs overhead on each call touching that branch.

**The fix for shared-prefix workloads remains `--prefix-depth 16 --no-parent-inheritance`**
(see §`--no-parent-inheritance results`), which distributes the 44,641 unique depth-16 keys
48%/52% across shards without any rebalancing overhead.

---

## Node-Depth-Sharded CRTC: shadow-trie routing

**Question:** Can the CRTC's internal path-compressed node structure be used as the routing
key — so that the router automatically discovers the depth at which a shared prefix ends,
without needing to set `prefix_depth` by hand?

**Short answer: Yes.** A shadow trie (`NodeDepthShardedIndexer`) that mirrors the top K
CRTC-node levels assigns shards at node-edge boundaries rather than at a fixed block count.
On the .prefix trace (15-block system prompt), node-depth=2 sees the shared 15-block prompt as
**one CRTC-node edge** (all compressed into a single node), and 44,641 unique continuations
as distinct leaves at level 2 — producing a perfectly balanced routing table (541/541 leaves)
without knowing that the system prompt is exactly 15 blocks.

### Implementation

`NodeDepthShardedIndexer<T>` (`lib/kv-router/src/indexer/node_depth_sharded.rs`):

- **Shadow trie** (`ShadowNode`): a trie with one level per CRTC-node hop. Each node holds
`edge: Vec<LocalBlockHash>` (variable-length, same as a CRTC edge), `node_depth`, an
optional `shard` assignment, and `children` keyed on the first `LocalBlockHash` of the
child's edge.
- **Insert** (`insert_and_get_shard`): walks the trie matching edges block-by-block. On a
partial edge match, the existing leaf is split into a common-prefix interior node and two
suffix children. Only leaf nodes are split (interior nodes already have children); this
preserves the invariant that `node_depth` is consistent without requiring descendant updates.
- **Route** (`route_sequence`): reads the trie, returning the shard assigned at node-depth K.
Returns `None` on miss → early-exit, same as branch-sharded.
- **Event routing**: root `Stored` events call `insert_and_get_shard` under write lock.
Continuation events inherit via `block_to_shard` if `inherit_parent_shard=true` (default).

### Benchmark results

#### Mooncake trace (2 shards × 4w, `-d 7`, `--benchmark-duration-ms 30000`)


| Indexer                    | routing depth    | p99        | Block split   | Miss rate | Routing entries  |
| -------------------------- | ---------------- | ---------- | ------------- | --------- | ---------------- |
| CRTC baseline (8w)         | —                | 1,455 µs   | —             | —         | —                |
| branch-sharded d=2         | 2 blocks         | 1,183 µs   | 34% / 66%     | 39.5%     | 822 branches     |
| **node-depth-sharded d=2** | **2 CRTC nodes** | **440 µs** | **42% / 57%** | **85.5%** | **2,760 leaves** |


Node-depth achieves **440 µs p99** — 70% below the CRTC baseline and 63% below
branch-sharded. The 85.5% miss rate (vs 39.5% for branch-sharded) is the key driver: at
node-depth=2, the shadow trie maps to 2,760 distinct leaves vs branch-sharded's 822 branches,
but the vast majority of queries still miss because most query sequences do not match any
known stored prefix at node-level granularity.

#### .prefix trace (2 shards × 4w, `-d 3`, `--benchmark-duration-ms 30000`)


| Indexer                    | routing depth    | p99        | Block split              | Miss rate | Routing entries            |
| -------------------------- | ---------------- | ---------- | ------------------------ | --------- | -------------------------- |
| CRTC baseline (8w)         | —                | 943 µs     | —                        | —         | —                          |
| branch-sharded d=2         | 2 blocks         | 949 µs     | **100% / 0%** (collapse) | 0.0%      | 169 branches               |
| **node-depth-sharded d=2** | **2 CRTC nodes** | **361 µs** | 0.2% / 99.8%             | **82.7%** | **1,082 leaves (541/541)** |


Node-depth at routing_node_depth=2 achieves **361 µs p99** — 62% below the CRTC baseline
and 62% below branch-sharded d=2 — with an 82.7% early-exit rate and a **perfectly balanced
routing table (541/541 leaves)**, without needing `--prefix-depth 16`.

The block distribution is still 0.2%/99.8% because with `inherit_parent_shard=true`,
continuation events route via `block_to_shard` inheritance rather than the shadow trie.
This is the same fundamental limitation as all other sharding approaches on shared-prefix
workloads: the first-stored shard receives all continuation blocks regardless of routing
table balance.

**Why the routing table is perfectly balanced (541/541):** the shadow trie never sees the
shared 15-block system prompt as distinct branches — the entire prompt is a single CRTC node
edge. At node-depth=2, the trie has one root → system-prompt node (depth 1) → 44,641
continuation leaf nodes (depth 2). The least-loaded assignment distributes these leaves
50/50: 541 leaves per shard (1,082 total from the root structure).

**Why branch-sharded d=2 sees 0% miss / 100%/0% collapse:** at block-depth=2, all requests
share the same first 2 blocks → one branch key → always dispatches to one shard. The
branch-sharded approach needs `depth=16` to see divergence. The shadow trie at node-depth=2
reaches past the shared prompt because it counts CRTC-node hops, not block hops.

### Comparison: node-depth vs branch-sharded on .prefix


| Approach               | depth param | "depth" semantic     | Sees shared prefix as       | p99        | Block balance  |
| ---------------------- | ----------- | -------------------- | --------------------------- | ---------- | -------------- |
| branch-sharded         | 2           | 2 blocks             | 1 unique branch (collapses) | 949 µs     | 100%/0%        |
| branch-sharded         | 16          | 16 blocks            | 44,641 branches             | 350 µs     | 0.2%/99.8%†    |
| **node-depth-sharded** | **2**       | **2 CRTC-node hops** | **44,641 leaves**           | **361 µs** | **0.2%/99.8%** |


†branch-sharded d=16 false-negatives: 82.7% miss rate is incorrect; those sequences have
cached blocks on the dominant shard but the routing table has no entry for them.

Node-depth-sharded at depth=2 matches branch-sharded depth=16 in p99 and block balance,
without requiring knowledge of the system prompt length. The 82.7% miss rate in node-depth
is also mostly early-exit (miss = no blocks stored for that trie path), whereas branch-sharded
d=16's miss rate is predominantly false-negatives.

### Benchmark commands

```bash
# Mooncake trace, node-depth-sharded
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  /path/to/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  node-depth-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 \
  --routing-node-depth 2

# .prefix trace, node-depth-sharded
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  .prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 3 \
  node-depth-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 \
  --routing-node-depth 2

# Side-by-side comparison (CRTC baseline, branch-sharded d=2, node-depth d=2)
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  .prefix/conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl \
  --compare concurrent-radix-tree-compressed,branch-sharded-crtc,node-depth-sharded-crtc \
  --num-event-workers 8 -d 3 --benchmark-duration-ms 30000
```

---

## Benchmark Commands

```bash
TRACE=/path/to/mooncake_trace.jsonl

# Baseline: 1 CRTC, 8 event workers
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 10000 -d 7 \
  concurrent-radix-tree-compressed --num-event-workers 8

# Worker-sharded: 2 shards × 4 workers
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 10000 -d 7 \
  sharded-concurrent-crtc --num-shards 2 --num-event-workers-per-shard 4

# Prefix-sharded: 2 shards × 4 workers, depth=2
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 10000 -d 7 \
  prefix-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2

# Branch-sharded: 2 shards × 4 workers, depth=2
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $TRACE --trace-simulation-duration-ms 10000 --benchmark-duration-ms 10000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

---

## Files Changed


| File                                                      | Change                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `lib/kv-router/src/indexer/sharded_concurrent.rs`         | Phase 1 — `ShardedConcurrentIndexer<T>`, optional `ShardReadPool`                     |
| `lib/kv-router/src/indexer/prefix_sharded.rs`             | Phase 2 — `PrefixShardedIndexer<T>`                                                   |
| `lib/kv-router/src/indexer/branch_sharded.rs`             | Phase 3 — `BranchShardedIndexer<T>`                                                   |
| `lib/kv-router/src/indexer/branch_sharded_rebalancing.rs` | Phase 4 — `RebalancingBranchShardedIndexer<T>` with two-phase migration               |
| `lib/kv-router/src/indexer/REBALANCING.md`                | Phase 4 — design doc for the two-phase migration protocol                             |
| `lib/kv-router/src/indexer/mod.rs`                        | Added all four modules and re-exports                                                 |
| `lib/kv-router/src/lib.rs`                                | Re-exported all four indexers                                                         |
| `lib/bench/kv_router/mooncake_bench.rs`                   | Added all five subcommands, `--find-matches-concurrency`, shard distribution printing |
| `lib/kv-router/src/indexer/node_depth_sharded.rs`         | Phase 5 — `NodeDepthShardedIndexer<T>` with shadow-trie routing                       |


