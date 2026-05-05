# Benchmarking the Sharded KV Router

## Trace Data

Benchmarks use JSONL trace files. Each line is a JSON object with fields:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | float (ms) | Absolute arrival time, or omit and use `delay` |
| `delay` | float (ms) | Time since previous request (alternative to `timestamp`) |
| `hash_ids` | `u64[]` | Block-level KV cache hash IDs |
| `output_length` | `u64` | Output token count |
| `input_length` | `u64` (optional) | Input token count |

### Public traces (Mooncake FAST25)

Three traces from the Mooncake FAST25 paper:

```bash
mkdir -p lib/kv-router/traces
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl \
  -o lib/kv-router/traces/conversation_trace.jsonl
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl \
  -o lib/kv-router/traces/synthetic_trace.jsonl
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o lib/kv-router/traces/toolagent_trace.jsonl
```

| File | Description |
|------|-------------|
| `conversation_trace.jsonl` | Conversational workload (diverse prefixes) |
| `synthetic_trace.jsonl` | Synthetic workload |
| `toolagent_trace.jsonl` | Agentic/tool-use workload |

---

## Two benchmark modes

**Steady-state** (`--benchmark-duration-ms 30000`): replays the trace at its natural arrival rate. Measures real-world p99 latency. Throughput will match the trace rate — both indexers should keep up, so ops/s will be similar; p99 is the meaningful comparison.

**Peak throughput sweep** (`--sweep`): progressively shrinks the benchmark window to drive offered rate above saturation. Use this to compare maximum throughput across indexers.

---

## Running the Benchmarks

All benchmarks run through `mooncake_bench` in `dynamo-bench`. **Run from the repository root.** The bench binary runs in `lib/bench/`, so trace paths must be absolute — the commands below use `$(git rev-parse --show-toplevel)` for portability.

```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  <TRACE_PATH> [global options] <INDEXER_SUBCOMMAND> [indexer options]
```

### Self-test (no trace required)

```bash
cargo test --package dynamo-bench --test mooncake_trace
```

### Steady-state (p99 at real-world request rate)

**CRTC baseline (8 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  concurrent-radix-tree-compressed --num-event-workers 8
```

**Branch-sharded depth=2 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

**Branch-sharded depth=4 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 4
```

**Anchor-aware branch-sharded depth=2 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  anchor-aware-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

### Peak throughput sweep

**CRTC baseline — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  concurrent-radix-tree-compressed --num-event-workers 8
```

**Branch-sharded depth=2 — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

**Anchor-aware branch-sharded depth=2 — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  anchor-aware-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

### Worker scaling

```bash
for factor in 1 2 4 8 16 32; do
  cargo bench --package dynamo-bench --bench mooncake_bench -- \
    $(git rev-parse --show-toplevel)/lib/kv-router/traces/conversation_trace.jsonl \
    --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 \
    --num-unique-inference-workers 1000 \
    --trace-duplication-factor $factor \
    -d 7 \
    branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
done
```

---

## Understanding the output

### Standard fields (all indexers)

| Field | Meaning |
|-------|---------|
| Offered ops/s | Planned request rate = total ops / benchmark window |
| Achieved ops/s | Actual completed rate — matches offered when not saturated |
| p99 latency | 99th-percentile `find_matches` latency |

### Branch-sharded extra fields

| Field | Meaning |
|-------|---------|
| Exact dispatch | Queries routed by the deepest requested prefix key |
| Shallow-fallback % | Queries whose exact requested prefix key missed but a shorter registered prefix routed to a shard that can return the shallow overlap score |
| Early-exit % | Queries with no registered prefix alias, resolved without shard dispatch |
| Avg routing | Prefix-key routing-table lookup time (deepest registered prefix → shard index) |
| Avg shard | CRTC traversal time on the dispatched shard |

---

## Key CLI Flags

| Flag | Default | What it does |
|------|---------|-------------|
| `-d N` | 1 | Worker duplication factor: replay the trace with N copies of each unique worker (higher = more write pressure) |
| `--find-matches-concurrency N` | 0 | N additional tokio tasks issuing `find_matches` in a tight loop alongside trace replay; stresses the read path |
| `--trace-simulation-duration-ms` | — | Rescale trace to this wall-clock duration (ms); omit to preserve original Mooncake timestamps |
| `--benchmark-duration-ms` | 60000 | Measurement window; shorter = higher offered rate |
| `--num-unique-inference-workers` | 1000 | Workers partitioned from the trace |
| `--trace-length-factor` | 1 | Stretch each request's hash sequence |
| `--trace-duplication-factor` | 1 | Create structurally identical copies with disjoint hash spaces |
| `--seed` | 42 | RNG seed for worker-to-trace assignment |
| `--sweep` | off | Find saturation point by varying benchmark window |
| `--sweep-min-ms` | 1000 | Shortest benchmark window in sweep |
| `--sweep-max-ms` | 50000 | Longest benchmark window in sweep |
| `--sweep-steps` | 10 | Number of steps in sweep |
| `--shard-metrics-csv FILE` | — | Sample shard block/node counts over time → CSV + SVG |

### `branch-sharded-crtc` flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--num-shards` | 2 | Number of independent CRTC shards |
| `--num-event-workers-per-shard` | 4 | OS threads per shard for KV event processing |
| `--prefix-depth` | 2 | Blocks hashed to compute the branch routing key |

### `anchor-aware-branch-sharded-crtc` flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--num-shards` | 2 | Number of independent CRTC shards |
| `--num-event-workers-per-shard` | 4 | OS threads per shard for KV event processing |
| `--prefix-depth` | 2 | Maximum routing-trie depth before dispatching to one shard |

Note: `anchor-aware-branch-sharded-crtc` does not support approximate pruning. It provides a stronger routing-correctness guarantee for out-of-order events, at the cost of higher routing latency and a hot-branch shard collapse risk on dominant-prefix workloads (see Known Issues).

---

## Results

Trace: `conversation_trace.jsonl` (Mooncake FAST25). Config: 2 shards × 4 workers for branch-sharded, 8 workers for CRTC baseline, `-d 7`.

### Steady-state — p99 at real-world request rate (~11,860 ops/s offered)

| Indexer | Achieved ops/s | p99 | Routing outcome | Avg routing | Avg shard |
|---------|---------------|-----|-----------------|-------------|-----------|
| CRTC baseline (8w) | 11,540 | 5,768 µs | — | — | — |
| Branch-sharded depth=2 (2×4w) | 11,846 | **769 µs** | 14.6% exact / 85.4% shallow-fallback / 0.0% miss | 498 ns | 161 µs |
| Branch-sharded depth=4 (2×4w) | 11,818 | **855 µs** | 13.0% exact / 87.0% shallow-fallback / 0.0% miss | 601 ns | 176 µs |
| Anchor-aware BSI depth=2 (2×4w) | 11,740 | 1,006 µs | 18.5% TRIE-only | 394 µs | 8 µs |

Branch-sharded depth=2 p99 is **7.5× lower** than CRTC; depth=4 is **6.7× lower**. Both branch-sharded runs effectively keep up with the offered trace rate. Routing remains sub-microsecond, while CRTC traversal on the selected shard dominates request latency.

The low true-miss rate is expected for this trace: most queries share a registered shallow prefix and are dispatched through shallow-fallback so the shard can return a shallow overlap score instead of an empty result. That keeps accuracy aligned with the unsharded CRTC baseline while preserving single-shard dispatch. In these runs, shallow-fallback accounts for **85.4%** of depth=2 lookups and **87.0%** of depth=4 lookups.

Anchor-aware BSI sits between CRTC and branch-sharded in p99 on this trace. Avg routing is 394 µs because the routing TRIE does more work before dispatch; avg shard time is only 8 µs because many blocks are represented by the routing TRIE and never reach the CRTC. Anchor-aware BSI provides a stronger routing-correctness guarantee for out-of-order events, but its routing overhead is materially higher on this workload.

> **Shard imbalance warning (this trace):** `conversation_trace.jsonl` has highly similar conversation prefixes (shared system prompts). With `prefix_depth` 2 and 4, branch-sharded routing put all stored blocks on one shard for the single-copy trace. This does not prevent the trace-rate latency win above, but it means these numbers do not show the ideal multi-shard scaling case. See Known Issues.

Shard block distribution:
```text
depth=2:  shard 0: 2,128,077 blocks (100.0%), 7,000 workers  shard 1: 0 blocks (0.0%), 0 workers
          branches: shard[0]=831, shard[1]=0

depth=4:  shard 0: 2,128,077 blocks (100.0%), 7,000 workers  shard 1: 0 blocks (0.0%), 0 workers
          branches: shard[0]=959, shard[1]=0
```

Increasing `prefix_depth` from 2 to 4 does not distribute blocks across shards on this trace, and this steady-state sample shows slightly higher p99 at depth=4. The common prefix is still too dominant for these depths. See Known Issues below.

### Peak throughput sweep — `conversation_trace.jsonl`

Peak is defined as the highest achieved ops/s before the bench warns it cannot keep up with the offered rate.

| Indexer | Peak achieved ops/s | p99 at peak | vs CRTC |
|---------|--------------------:|-------------|---------|
| CRTC baseline (8w) | 18,046 | 13,362 µs | — |
| Branch-sharded depth=2 (2×4w) | **130,096** | **655 µs** | **+621%, 20× lower p99** |

CRTC saturates at ~18k ops/s with p99 exceeding 10,000 µs at all higher offered rates. Branch-sharded sustains ~130k ops/s before the first bench warning, with p99 under 700 µs at that point. The gain comes from cheap prefix routing plus one-shard CRTC traversal; it does not depend on early exits.

Full sweep data:

**CRTC baseline:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 30,000 ms | 11,861 | 11,532 | 7,536 µs |
| 18,455 ms | 19,281 | 18,046 | 13,362 µs |
| 11,352 ms ⚠ | 31,345 | 25,240 | 11,076 µs |
| 6,983 ms ⚠ | 50,956 | 30,547 | 10,529 µs |
| 4,296 ms ⚠ | 82,827 | 25,050 | 11,607 µs |
| 2,643 ms ⚠ | 134,629 | 23,684 | 10,233 µs |
| 1,626 ms ⚠ | 218,834 | 27,543 | 11,143 µs |
| 1,000 ms ⚠ | 355,824 | 25,249 | 12,388 µs |

**Branch-sharded depth=2:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 18,455 ms | 19,281 | 19,231 | 529 µs |
| 11,352 ms | 31,345 | 31,250 | 591 µs |
| 6,983 ms | 50,956 | 50,727 | 511 µs |
| 4,296 ms | 82,827 | 82,224 | 531 µs |
| 2,643 ms | 134,629 | 130,096 | 655 µs |
| 1,626 ms ⚠ | 218,834 | 183,918 | 1,338 µs |
| 1,000 ms ⚠ | 355,824 | 217,263 | 1,453 µs |

**Anchor-aware BSI depth=2:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 30,000 ms | 11,861 | 11,797 | 2,069 µs |
| 18,455 ms | 19,281 | 18,777 | 1,454 µs |
| 11,352 ms | 31,345 | 30,778 | 1,376 µs |
| 6,983 ms | 50,956 | 47,684 | **1,228 µs** |
| 4,296 ms | 82,827 | 77,039 | 1,496 µs |
| 2,643 ms ⚠ | 134,629 | 114,488 | 1,984 µs |
| 1,626 ms ⚠ | 218,834 | 114,587 | 2,042 µs |
| 1,000 ms ⚠ | 355,824 | 106,663 | 2,427 µs |

Anchor-aware BSI saturates at ~77k ops/s (no warning at 4,296 ms; first warning at 2,643 ms). Best p99 is 1,228 µs at moderate load, vs 511-655 µs for branch-sharded in the same offered-rate range. The TRIE routing overhead sets a floor: avg routing is ~394 µs regardless of throughput. The caveat on this trace (single hot shard) means these numbers do not represent ideal anchor-aware performance — on a trace with more diverse prefixes the shard load would distribute.

⚠ = bench warned it could not keep up with the offered rate.

### Worker scaling — branch-sharded depth=2

Tests how p99 and shard balance change as the number of inference workers grows.
Config: 2 shards × 4 workers per shard, `--num-unique-inference-workers 1000`, `-d 7` (7 replicas × 1,000 workers = 7,000 concurrent workers), `--benchmark-duration-ms 30000`.

| Duplication factor | Effective workers | Branches | Offered ops/s | Achieved ops/s | Early-exit | Avg routing | Avg shard | p99 | Block split |
|-------------------|------------------:|---------:|--------------:|---------------:|------------|-------------|-----------|-----|-------------|
| 1× | 7,000 | 831 | 11,860 | 11,846 | 0.0% | 594 ns | 192 µs | 1,301 µs | 100.0% / 0.0% |
| 2× | 14,000 | 1,650 | 23,721 | 23,670 | 0.0% | 472 ns | 176 µs | 505 µs | 50.0% / 50.0% |
| 4× | 28,000 | 3,329 | 47,443 | 47,370 | 0.0% | 499 ns | 157 µs | **464 µs** | 50.0% / 50.0% |
| 8× | 56,000 | 6,646 | 94,886 | 94,749 | 0.0% | 425 ns | 174 µs | 811 µs | 50.0% / 50.0% |
| 16× | 112,000 | 13,296 | 189,772 | 184,034 | 0.0% | 432 ns | 185 µs | 1,363 µs | 37.5% / 62.5% |
| 32× ⚠ | 224,000 | 26,617 | 379,543 | 160,396 | 0.0% | 550 ns | 227 µs | 2,121 µs | 50.0% / 50.0% |

**1× is the most skewed case.** The single-copy trace lands entirely on one shard for `prefix_depth=2`, so p99 reflects one-shard CRTC traversal rather than multi-shard load distribution.

**2× to 8× show the intended balanced case.** Once duplicated hash spaces introduce more distinct branch keys, block distribution is near 50/50 and p99 stays at or below 811 µs while offered throughput scales from 23.7k to 94.9k ops/s.

**16× remains usable but starts to show pressure.** It keeps up with the offered rate within a few percent, but p99 rises to 1,363 µs and block distribution skews to 37.5%/62.5%.

**32× is outside the clean measurement range for this command.** The bench warns that it cannot keep up and macOS reports allocation pressure, so the 160k achieved ops/s is not a reliable indexer ceiling. Avg routing is still sub-microsecond at 26,617 branches, which suggests the routing map is not the limiting factor.

**Practical implication:** For this trace and a 2-shard config, the cleanest latency/throughput region is 14k-56k effective workers. Higher worker counts need either a longer benchmark window, more memory headroom, or more shards to separate indexer limits from benchmark-driver pressure.

---

## Known Issues

### Shared-prefix shard collapse and lifetime block skew

`assign_shard` uses live block count as the primary load metric, so branch placement adapts to observed load. Two skew modes remain:

1. **Shared-prefix collapse:** if many requests share the same first `prefix_depth` blocks, they share the same routing aliases and land on one shard. Observed on the single-copy `conversation_trace.jsonl` run: 831-959 branch aliases and 2,128,077 blocks all landed on shard 0.
2. **Lifetime skew:** even when branch counts are distributed, branches are placed permanently and some conversations accumulate far more blocks than others. Over time, a few heavy branches can dominate one shard.

The hot shard's larger tree drives up p99.

**Future work — rebalancing:** periodically migrate heavy branches from the overloaded shard to lighter ones. This directly addresses lifetime skew when branches can be moved whole. It does not fully solve shared-prefix collapse (see `prefix_depth` section below), where the routing key itself is too coarse and a structural fix like node-depth routing is needed instead.

### `prefix_depth` must be tuned per workload

If most requests share a long system prompt, all conversations may hash to the same first `prefix_depth` blocks → single branch key → one shard gets all traffic. Set `prefix_depth` to span the shared prefix plus at least 1–2 unique blocks.

Note: `anchor-aware-branch-sharded-crtc` avoids FNV routing-key collisions (different conversations with the same prefix still get distinct TRIE paths) but has its own hot-branch collapse issue on dominant-prefix workloads (see below). Neither variant is unconditionally better here — tune `prefix_depth` regardless of which you use.

### Anchor-aware BSI: hot-branch shard collapse

`AnchorAwareBranchShardedIndexer` uses static divergent-shard assignment: the first conversation under a new parent node stays on the parent's shard; only subsequent divergent siblings are hashed to a different shard. On workloads with a dominant shared prefix (e.g. one system prompt used by >99% of conversations), nearly all traffic ends up as the "first child" of the same TRIE node and routes to one shard. Observed on `conversation_trace.jsonl`: 100% of blocks on shard 1.

The code has an open TODO for adaptive hot-branch splitting. Until that is resolved, `branch-sharded-crtc` (with sticky routing) is the safer choice for traces with narrow prefix diversity.
