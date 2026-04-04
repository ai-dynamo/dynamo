# Branch-Sharded CRTC vs Single CRTC — Benchmark Comparison

**Date:** 2026-04-02
**Config:** 4 shards × 4 event-workers (branch-sharded); 16 or 4 event-workers (single CRTC)
**Traces:**
- `.jcs` — conversational synthetic trace (`conversation_trace_synth_15x1+10.0_speedup1_maxisl163840.jsonl`), shared 15-block system prompt
- `mooncake` — mooncake production trace (`mooncake_trace.jsonl`)

---

## Results

### JCS trace (prefix_depth=17 for branch-sharded)

| Indexer                            | Event workers | Achieved ops/s | Block ops/s | p99 latency |
|------------------------------------|--------------|----------------|-------------|-------------|
| branch-sharded-crtc (depth=17)     | 4×4 = 16     | 187,187        | 628,247     | **87 µs**   |
| concurrent-radix-tree-compressed   | 16           | 187,181        | 628,244     | 254 µs      |
| concurrent-radix-tree-compressed   | 4            | 187,188        | 628,248     | 234 µs      |

**Shard balance (branch-sharded):**
```
shard[0]: 12,931 branches, 1,070,956 blocks (25.8%), 316,087 nodes
shard[1]: 12,931 branches, 1,035,110 blocks (25.0%), 310,694 nodes
shard[2]: 12,931 branches, 1,021,227 blocks (24.6%), 300,091 nodes
shard[3]: 12,930 branches, 1,018,430 blocks (24.6%), 304,422 nodes
```

**find_matches routing (branch-sharded):**
```
100,000 total: 16,708 dispatched, 83,292 early-exit (83.3% miss)
avg routing    = 745 ns   (branch_to_shard lookup)
avg shard      = 56 µs    (CRTC traversal, inline)
remove broadcasts = 777,729
```

---

### Mooncake trace (prefix_depth=2 for branch-sharded)

| Indexer                            | Event workers | Achieved ops/s | Block ops/s | p99 latency |
|------------------------------------|--------------|----------------|-------------|-------------|
| branch-sharded-crtc (depth=2)      | 4×4 = 16     | 1,197          | 11,879      | **305 µs**  |
| concurrent-radix-tree-compressed   | 16           | 1,197          | 11,879      | 385 µs      |
| concurrent-radix-tree-compressed   | 4            | 1,197          | 11,879      | 295 µs      |

**Shard balance (branch-sharded):**
```
shard[0]: 1,749 branches,  34,251 blocks (27.1%),  6,451 nodes
shard[1]: 1,749 branches,  22,224 blocks (17.6%),  2,880 nodes
shard[2]: 1,749 branches,  21,962 blocks (17.4%),  3,017 nodes
shard[3]: 1,749 branches,  48,040 blocks (38.0%), 11,813 nodes
```

**find_matches routing (branch-sharded):**
```
23,608 total: 13,630 dispatched, 9,978 early-exit (42.3% miss)
avg routing    = 258 ns   (branch_to_shard lookup)
avg shard      = 88 µs    (CRTC traversal, inline)
remove broadcasts = 0
```

---

## Analysis

### JCS trace: branch-sharded wins clearly (87 µs vs 234–254 µs, ~3× lower p99)

The 83% early-exit rate drives most of the gain. This trace has a shared 15-block system
prompt followed by unique conversation content. With prefix_depth=17, the branch key spans
the full system prompt + first 2 unique blocks, giving a distinct key per conversation.
Queries for unseen conversations short-circuit in ~745 ns without touching any shard.

Branch and block balance are near-perfect (≤1.2% deviation) because the 17-block prefix
captures enough entropy from the unique conversation content to spread branches uniformly
across shards by the least-loaded assignment.

Note: CRTC with 4 workers (234 µs) is marginally faster than 16 workers (254 µs) on this
trace — fewer workers means less lock contention on the single shared tree.

### Mooncake trace: branch-sharded wins on p99 vs CRTC-16, ties CRTC-4 (305 vs 295 µs)

Block distribution is significantly skewed: shard[3] holds 38% of blocks vs 17% on
shards[1,2]. Branch counts are perfectly balanced (1,749 each) because the least-loaded
balancer counts *branches*, not *blocks* — but some branches in this trace accumulate far
more blocks than others (high-reuse sequences). This hot-shard effect drives up shard[3]'s
p99 and eliminates the latency advantage.

With depth=2, the 42% early-exit rate is lower (compared to 83% on JCS) because the
mooncake trace has more distinct 2-block prefixes that are known, so more queries dispatch.

### Takeaways

1. **Branch-sharded depth=17 on .jcs is the standout result**: 3× p99 improvement over
   any CRTC configuration, near-perfect balance. The high early-exit rate on conversational
   traces is a structural win — unknown branches are free.

2. **Mooncake branch balance is broken by block skew**: least-loaded-by-branch-count does
   not account for variable branch depth. A block-count–weighted balancer or the
   rebalancing variant would fix shard[3]'s 38% overload.

3. **CRTC worker count has marginal effect on throughput** (both traces saturate at the
   same offered rate) but small effect on p99: 4 workers ≤ 16 workers for this load level,
   suggesting the write-path lock is not the bottleneck at this concurrency.
