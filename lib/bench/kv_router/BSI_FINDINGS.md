# Branch-Sharded Indexer Findings

This is the current state of the Branch-Sharded Indexer (BSI): what Janelle's original work showed, why the first fast shape was not correct, what changed as correctness work landed, what the current benchmarks say now, and what looks most promising next.

The short version: BSI moved from "very fast but not correct enough" to "correct and balanced, but still not consistently faster than non-sharded CRTC." The remaining problem is not just shard placement. The hard part is preserving exact prefix correctness without eagerly building large worker score maps on the read path.

## Timeline

| Step | What changed | Why it mattered |
|------|--------------|-----------------|
| Janelle's original BSI | Flat branch routing with a cheap one-shard lookup and an empty-result early exit | Showed the performance upside of sharding, but skipped state needed for exact prefix correctness |
| Anchor-aware BSI | Router-owned shallow trie, shard anchors, parent-liveness filtering, boundary fallback | Fixed the major false-miss and zombie-descendant classes, but made reads more expensive |
| Scheduler-hop removal | Removed a scheduler hop from BSI shard reads | Reduced async scheduling overhead on the dispatched-read path |
| Worker cleanup indexes | Replaced broad BSI cleanup scans with worker indexes | Reduced cleanup/reconciliation overhead and made removal bookkeeping less scan-heavy |
| Borrowed shard-read suffixes | Borrowed suffix slices instead of allocating a suffix `Vec` per dispatched read | Narrow hot-path allocation win; earlier stress runs showed a large p99 improvement for BSI |
| Current code (in a branch) | Deterministic child shard placement, prefix-boundary fallback correctness, single-pass active reconciliation, replay/zombie tests | Keeps the corrected semantics while removing the old all-block shard collapse seen on this trace |

## Janelle's Fast Numbers

Janelle's original BSI shape was compelling because it approximated the ideal sharded read path:

```text
compute branch key
look up branch -> shard
query exactly one CRTC shard
return scores
```

When the branch key was unknown, it could return an empty result immediately. That early-exit path was extremely cheap and explains much of the win.

Representative Janelle-era numbers:

| Workload | Indexer | Achieved ops/s | p99 | Notes |
|----------|---------|---------------:|----:|-------|
| JCS conversational trace, depth=17 | BSI 4x4 | 187,187 | 87 us | 83.3% early exit; near-perfect shard balance |
| JCS conversational trace, depth=17 | CRTC 16w | 187,181 | 254 us | Same offered load |
| Mooncake trace, depth=2 | BSI 4x4 | 1,197 | 305 us | 42.3% early exit |
| Mooncake trace, depth=2 | CRTC 16w | 1,197 | 385 us | Same offered load |
| Mooncake sweep, depth=2 | BSI 2x4 | 376,465 | 1,795 us | +73% achieved ops/s vs CRTC in that older sweep |
| Mooncake sweep | CRTC 8w | 218,013 | 1,941 us | Older sweep baseline |

Those numbers are real and useful as a performance target. They are not a correctness baseline.

## Why Old BSI Was Not Correct

The old flat branch-table design did less work because it lost information that exact prefix matching needs.

| Issue | Why it matters |
|-------|----------------|
| Short/shallow query false misses | A stored branch keyed at `prefix_depth` can exist even when a query only reaches a shallower prefix. Treating "unknown exact branch key" as "no match" silently drops valid prefix scores. |
| Continuation routed away from parent | A continuation can hash to a different shard than the parent chain. If the new shard does not have the parent prefix, shard-local lookup can drop the continuation. |
| Zombie descendants after parent removal | If a parent prefix is removed but a descendant is still present, the descendant must not score beyond the removed parent. A shard-local suffix lookup cannot know that by itself. |
| Boundary-prefix workers | A worker with only the prefix up to the routing boundary still needs a boundary score when the query continues and the suffix misses. |
| Dump/replay mismatch | Dumping only shard-local events does not preserve routing state, shallow prefix state, or parent-liveness semantics needed for replay-equivalent scores. |

The old design was therefore a good performance sketch, but correctness required bringing parent context back into the query path.

## Current Correct BSI

The current BSI is anchor-aware. It keeps a shallow routing trie, installs anchors into shards, and uses router-owned prefix state to keep shard suffix results honest.

```text
walk router-owned prefix
track live workers through parent nodes
dispatch suffix lookup from an anchor to one shard
filter shard suffix scores through the active parent set
preserve router-only fallback scores at the boundary
```

That fixes the silent false-miss and zombie-descendant classes, but it makes reads more expensive than the old branch-table path. The router now does exact worker-set work before and after shard lookup.

## What Improved

| Improvement | Effect | Evidence |
|-------------|--------|----------|
| Anchor-aware correctness | Makes BSI exact across shallow queries, parent removal, continuation routing, and replay | Focused `branch_sharded` tests cover boundary workers, parent removal, zombie descendants, and dump/replay equivalence |
| Deterministic child placement | Hashes every divergent child by sequence instead of letting the first child inherit the parent shard | Current Mooncake depth=3 block split is stable around `55.5% / 44.5%`; old observed runs collapsed as far as effectively `0% / 100%` |
| Scheduler-hop removal | Avoids one dispatch/scheduling step in shard reads | Included in all current reruns |
| Worker cleanup indexes | Avoids broad cleanup scans during removal/reconciliation | Included in all current reruns |
| Borrowed suffixes | Avoids a suffix allocation per dispatched read | Earlier duplicated-Mooncake stress runs showed BSI p99 improving from `5,822 us` to `2,072 us` |
| Single-pass active reconciliation | Keeps exact parent-liveness filtering while avoiding extra dropped-worker collection | Covered by current correctness tests; included in current reruns |

## Benchmark Results

All current reruns below use `mooncake_trace.jsonl`, `--trace-simulation-duration-ms 10000`, `-d 7`, BSI `2 shards x 4 workers, prefix_depth=3`, and CRTC `8 workers`.

### Natural Mooncake Replay

| Scenario | CRTC 8w | BSI 2x4 depth=3 | Read |
|----------|--------:|----------------:|------|
| 30s steady, 5 runs, ~17,268 offered ops/s | 17,217 ops/s, p99 1,538 us | 16,806 ops/s, p99 4,830 us | Both keep up; CRTC has lower p99 |
| 10s pressure, 3 runs, ~51,805 offered ops/s | 51,378 ops/s, p99 1,339 us, no warnings | 38,769 ops/s, p99 6,853 us, warnings | BSI saturates even with stable shard split |
| Highest clean targeted point | 143,175 ops/s at 148,014 offered, p99 896 us | 41,429 ops/s at 43,171 offered, p99 3,520 us | CRTC clean ceiling is about 3.5x higher by achieved ops/s |
| First unstable/overloaded bracket | 151,198 ops/s at 160,287 offered, below 95% keep-up | unstable by 45,048 offered; two reruns warned | The BSI knee is sharp around 43k-45k offered ops/s |

The full BSI sweep had a couple of contradictory rows, so targeted reruns supersede the noisy sweep for the clean-ceiling claim. The stable BSI shard distribution in these runs was about `55.5% / 44.5%`, so the remaining gap is not the old "everything landed on one shard" failure mode.

### Duplicated-Mooncake Stress

An earlier 30s duplicated-Mooncake stress run showed the borrowed-suffix read-path change helping BSI:

| Target | Warnings | Achieved ops/s mean | p99 mean |
|--------|---------:|--------------------:|---------:|
| CRTC 8w | 0 | 34,379 | 6,238 us |
| BSI before borrowed suffixes | 0 | 33,758 | 5,822 us |
| BSI with borrowed suffixes | 0 | 34,216 | 2,072 us |

The same duplicated stress rerun on current main plus the current BSI changes looks different:

| Target | Warnings | Achieved ops/s mean | Block ops/s mean | p99 mean |
|--------|---------:|--------------------:|-----------------:|---------:|
| CRTC 8w | 0 | 34,447 | 337,221 | 1,094 us |
| BSI 2x4 depth=3 | 0 | 34,225 | 335,050 | 2,215 us |

The current BSI rerun still keeps up at ~34.5k offered ops/s and avoids the old ~5-6 ms BSI tail. The current same-machine CRTC run is also strong, so the conclusion is narrower: borrowed suffixes helped BSI, but the current exact CRTC baseline still wins this duplicated target.

## Why We Are Not Matching Janelle's Performance

Janelle's numbers came from two properties the correct implementation cannot keep exactly as-is:

1. Unknown branches could early-exit as empty. That is fast, but it is a false miss when shallow prefix state should score.
2. A shard could answer mostly from local state. That is fast, but it lacks enough parent-liveness context to handle continuation routing, parent removal, and boundary fallback exactly.

Correct BSI now does this:

```text
router prefix walk
active worker tracking
single-shard suffix lookup
parent-liveness filtering
fallback score materialization
score merge
```

Non-sharded CRTC has a simpler hot path:

```text
compressed trie lookup
return scores
```

The expensive part is exact worker-set handling. Hot prefixes can have thousands of live workers. If BSI eagerly writes fallback scores for all of them, the sharding win can disappear before the shard lookup has a chance to help.

## Old BSI vs Anchor-Aware BSI

| Design | Pros | Cons | Best use |
|--------|------|------|----------|
| Old flat BSI | Fast route plus one shard lookup; early-exit is nearly free; excellent numbers when branch keys align with the workload | Silent false misses; continuation/parent incoherence; weak replay semantics; no exact zombie-descendant guard | Performance reference, not safe as-is |
| Current anchor-aware BSI | Correct parent-liveness model; boundary fallback; replayable router state; supports remote shard handles | More read-path work; can materialize large score maps in hot-prefix cases; not always faster than CRTC | Current correctness baseline |
| Old BSI with correctness patches only | Tempting because it keeps the fast path for easy branches | The missing information has to live somewhere; bolting it on tends to recreate anchor-aware costs | Risky unless scoped to certified-safe branches |

## Better Sharding Direction

BSI may not be the right sharding axis for an exact indexer. Prefix sharding tries to make each query hit one shard, but exact prefix correctness keeps pulling parent context and fallback scores back into the router. That erodes the main benefit of sharding.

The more defensible multi-process direction is **worker-sharded exact CRTC with conservative shard summaries**.

```text
write path:
  route each worker's events to exactly one owner shard
  each shard maintains a complete exact CRTC for its workers

read path:
  coordinator consults conservative per-shard prefix summaries
  query only shards that can still produce a competitive exact score
  fall back to querying all shards when summaries are broad or uncertain
  merge exact per-worker scores
```

This preserves exactness because each worker's chain lives wholly inside one CRTC shard. There is no prefix-boundary parent problem, no zombie descendant split across shards, and no need to reconstruct global parent liveness in the router. The summary layer must be conservative: false positives are acceptable because they only add shard queries; false negatives are not allowed.

Why it can beat single-process CRTC:

| Property | Impact |
|----------|--------|
| Worker ownership is exact and stable | Writes scale across processes without duplicating every event |
| Per-shard CRTC stays unchanged | Reuses the current high-performance exact data structure |
| Queries run in parallel | Heavy lookups can use multiple processes instead of one process/thread pool |
| Conservative summaries can reduce fanout | If only a few shards can beat the current best score, the coordinator avoids all-shard broadcast |
| Fallback is exact | When summaries cannot prune safely, query all shards and merge |

This is still not a guaranteed win. If every query has to fan out to every shard and return thousands of scores, IPC and merge cost can lose to one in-process CRTC. The reason this path is still preferable to more BSI work is that the failure mode is measurable and safe: it degrades to exact scatter/gather, not silent correctness loss or complex prefix repair.

## Further Ideas Worth Keeping

| Idea | Why it could help | Guardrail |
|------|-------------------|-----------|
| Worker-sharded exact CRTC with conservative summaries | Enables multi-process scale while keeping each shard's CRTC exact | If summaries cannot prune safely, query all shards |
| Hybrid dense/sparse worker sets | Replaces hash-set-heavy reconciliation with word-parallel operations only for hot/high-cardinality sets | Use dense bitsets only above a measured cardinality threshold; otherwise keep sparse sets |
| Query fanout budget with auto-disable | Prevents summary routing from adding overhead when most shards remain candidates | If candidate shard count or merge cost crosses a threshold, bypass summaries for that workload |

## Benchmarking/measuring

Add or collect:

- active workers collected at the routing boundary
- workers scanned during reconciliation
- workers dropped during reconciliation
- fallback workers materialized at dispatch
- shard scores returned
- shard scores filtered out by parent liveness
- score merge/fallback time
- suffix length at dispatch
- routed shard block/worker distribution
- candidate shard count after summary pruning
- shard RPC/IPC latency
- exact score merge time across shards
- summary false-positive rate

Then run:

| Benchmark | Why |
|-----------|-----|
| Natural Mooncake steady, 5 runs | Baseline trace-rate behavior |
| Natural Mooncake pressure targets, 3-5 runs | Saturation and overload shape |
| Targeted clean-ceiling brackets | Avoid relying on noisy sweep rows |
| Duplicated Mooncake sustained stress | Compare against the best observed BSI p99 improvement |
| Config sweep: `2:4:2`, `2:4:3`, `2:4:4`, `4:2:3`, `8:1:3` | Catch depth and shard-count sensitivity |
| JCS/conversation trace | Re-check the workload where old BSI looked best |

The main success criterion should be a better BSI/CRTC ratio, not only an absolute BSI p99 move. Absolute p99 can drift with machine load; same-run ratios tell us whether BSI itself improved.
