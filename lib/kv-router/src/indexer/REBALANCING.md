# RebalancingBranchShardedIndexer — Design & Implementation Summary

## What problem does this solve?

`BranchShardedIndexer` assigns each new branch (identified by the FNV-1a hash of the first
`prefix_depth` block hashes) to the shard with the **fewest branches** at assignment time.  This
balances structural load (number of distinct prefix paths per shard), but not *query* load.

If one branch (e.g. a dominant system-prompt prefix) handles 90 % of all `find_matches` traffic,
the shard that owns it is permanently hot regardless of how many other branches exist.

`RebalancingBranchShardedIndexer` detects this skew and migrates the hottest branch to a cooler
shard at runtime.

---

## Files

| File | Role |
|------|------|
| `branch_sharded.rs` | Baseline (no rebalancing overhead) — unchanged |
| `branch_sharded_rebalancing.rs` | New file: `RebalancingBranchShardedIndexer<T>` |

Both implement the same `KvIndexerInterface` trait and can be compared directly in benchmarks:

```
mooncake_bench --compare branch-sharded-crtc,rebalancing-branch-sharded-crtc \
               --rebalance-interval-secs 10 \
               --imbalance-threshold 1.5 \
               --dual-write-window-secs 30
```

---

## What was added vs the baseline

| Addition | Purpose |
|----------|---------|
| `shard_query_counts: Vec<AtomicU64>` | Per-shard cumulative `find_matches` hits; drives rebalancer trigger |
| `branch_query_counts: DashMap<u64, AtomicU64>` | Per-branch hit counter; identifies *which* branch to migrate |
| `replaying_branches: DashSet<u64>` | Marks branches in Phase 1 — prevents premature dual-write |
| `dualwrite_branches: DashMap<u64, DualWriteEntry>` | Marks branches in Phase 2 — activates dual-write + scatter-gather |
| `rebalancer_handle: Mutex<Option<AbortHandle>>` | Allows `shutdown()` to stop the background task |
| `start_rebalancer(arc, interval, threshold, window)` | Spawns the periodic background task |
| `rebalance_once(threshold, window)` | One rebalance check; also callable in tests |
| `migrate_branch(key, old, new, window)` | Full two-phase migration implementation |
| `dispatch_find_matches(shard, seq)` | Helper: routes to read pool or inline (deduplicates code) |
| `merge_overlap_scores(a, b)` | Merges two `OverlapScores` from different shards (max per worker) |

---

## Two-phase migration protocol

### Why two phases?

A naive "dump → replay → switch" approach has a **race window**: events arriving between the dump
and the routing switch go to old_shard only.  After the switch, `find_matches` queries new_shard
and misses those blocks.

The fix is **dual-write + scatter-gather** during migration.  But enabling dual-write naively
causes an **ordering problem**: the CRTC drops continuation blocks whose parent is not yet stored
(`KvCacheEventError::ParentBlockNotFound` in `concurrent_radix_tree_compressed.rs`).  If a live
dual-write continuation arrives at new_shard before its root is replayed, it is permanently lost.

### The FIFO ordering guarantee

`ThreadPoolIndexer` routes events for each `WorkerId` to the same OS thread (sticky assignment)
via a **flume channel (FIFO)**.  This gives a key property:

> If replay events for worker W are *enqueued into new_shard's channel* **before** dual-write is
> activated, flume guarantees they are *processed* before any subsequent dual-write event for W.

### Phase 1 — Replaying

```
New Stored events  →  old_shard only   (dual-write NOT yet active)
find_matches       →  old_shard only

Migration task:
  1. replaying_branches.insert(branch_key)
  2. dump_events(old_shard)              [FIFO barrier: all in-flight writes drain]
  3. filter Stored events for branch_key
  4. for each event: shards[new_shard].apply_event(event)
     ↑ non-blocking send into flume channels — enqueues BEFORE any dual-write events
```

### Phase 2 — DualWrite  (activated atomically after step 4)

```
New Stored events  →  old_shard AND new_shard
find_matches       →  scatter-gather both, merge OverlapScores

Migration task:
  5. dualwrite_branches.insert(branch_key, {old, new})
  6. replaying_branches.remove(branch_key)   [Phase 2 now visible to apply_event]
  7. flush(new_shard)                        [wait for replay + early dual-writes]
  8. sleep(dual_write_window)                [gap-period traffic warms new_shard]
  9. branch_to_shard[branch_key] = new_shard
 10. dualwrite_branches.remove(branch_key)   [Phase 2 ends; single-shard resumes]
```

### Gap events

Events arriving between step 2 (dump) and step 6 (dual-write activation) go to old_shard only.
During the dual-write window (steps 7–10), those sequences generate new traffic that is
dual-written, so new_shard warms up naturally.  After step 10, new_shard handles the branch alone.

---

## OverlapScores merge

During Phase 2, `find_matches` queries both shards concurrently and merges results:

- **scores**: `max(old, new)` per `WorkerWithDpRank` (more prefix cached = better match)
- **tree_sizes**: `max(old, new)` per worker
- **frequencies**: element-wise `max` (both track the same query's access pattern)

---

## Bench CLI flags (RebalancingBranchShardedCrtc subcommand)

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-shards` | 2 | Number of CRTC shards |
| `--num-event-workers-per-shard` | 4 | OS write threads per shard |
| `--prefix-depth` | 2 | Branch key prefix depth |
| `--num-read-threads-per-shard` | 0 | 0 = inline reads |
| `--no-parent-inheritance` | false | Always route by branch key |
| `--rebalance-interval-secs` | 10 | How often to check for imbalance |
| `--imbalance-threshold` | 1.5 | max/avg ratio to trigger migration |
| `--dual-write-window-secs` | 30 | Duration of Phase 2 dual-write |
