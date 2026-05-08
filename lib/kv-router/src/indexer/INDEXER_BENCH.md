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
| Early-exit % | Queries resolved in ~300 ns with no shard dispatch (unknown branch key) |
| Avg routing | Routing-table lookup time (branch key → shard index) |
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

---

## Results

Trace: `conversation_trace.jsonl` (Mooncake FAST25). Config: 2 shards × 4 workers for branch-sharded, 8 workers for CRTC baseline, `-d 7`.

### Steady-state — p99 at real-world request rate (~11,860 ops/s offered)

| Indexer | Achieved ops/s | p99 | Early-exit | Avg routing | Avg shard |
|---------|---------------|-----|------------|-------------|-----------|
| CRTC baseline (8w) | 11,540 | 5,768 µs | — | — | — |
| Branch-sharded depth=2 (2×4w) | 11,706 | **1,387 µs** | 85.4% | 433 ns | 521 µs |
| Branch-sharded depth=4 (2×4w) | 11,775 | **727 µs** | 87.0% | 299 ns | 397 µs |

Branch-sharded depth=2 p99 is **4.2× lower** than CRTC; depth=4 is **7.9× lower**. The deeper key resolves more unique branches (1,038 vs 831), raising the early-exit rate slightly and reducing average shard traversal time.

Shard block distribution:
```text
depth=2:  shard 0: 1,061,550 blocks (91.1%), 3,556 workers  shard 1: 104,300 blocks  (8.9%), 3,444 workers
          branches: shard[0]=415, shard[1]=416  ← balanced branch count, 10:1 block skew

depth=4:  shard 0: 1,010,142 blocks (90.7%), 3,367 workers  shard 1: 103,222 blocks  (9.3%), 3,633 workers
          branches: shard[0]=416, shard[1]=622  ← unbalanced branch count, similar 10:1 block skew
```

The block skew (~10:1) persists at both depths despite different branch count distributions — confirming this is a lifetime skew problem (some branches inherently accumulate far more blocks), not an artifact of the assignment criterion. See Known Issues below.

### Peak throughput sweep — `conversation_trace.jsonl`

Peak is defined as the highest achieved ops/s before the bench warns it cannot keep up with the offered rate.

| Indexer | Peak achieved ops/s | p99 at peak | vs CRTC |
|---------|--------------------:|-------------|---------|
| CRTC baseline (8w) | 18,046 | 13,362 µs | — |
| Branch-sharded depth=2 (2×4w) | **122,585** | **583 µs** | **+579%, 23× lower p99** |

CRTC saturates at ~18k ops/s with p99 exceeding 10,000 µs at all higher offered rates. Branch-sharded sustains 122k+ ops/s with p99 under 600 µs — the 85% early-exit rate means most queries never touch a shard at all, so it scales far better under load.

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
| 30,000 ms | 11,861 | 11,770 | 885 µs |
| 18,455 ms | 19,281 | 19,052 | 794 µs |
| 11,352 ms | 31,345 | 30,561 | 799 µs |
| 6,983 ms | 50,956 | 49,033 | 746 µs |
| 4,296 ms | 82,827 | 76,323 | 598 µs |
| 2,643 ms | 134,629 | 122,585 | 583 µs |
| 1,626 ms ⚠ | 218,834 | 182,005 | 569 µs |
| 1,000 ms ⚠ | 355,824 | 255,190 | 573 µs |

⚠ = bench warned it could not keep up with the offered rate.

### Worker scaling — branch-sharded depth=2

Tests how p99 and shard balance change as the number of inference workers grows.
Config: 2 shards × 4 workers per shard, `--num-unique-inference-workers 1000`, `-d 7` (7 replicas × 1,000 workers = 7,000 concurrent workers), `--benchmark-duration-ms 30000`.

| Duplication factor | Effective workers | Branches | Offered ops/s | Early-exit | Avg routing | Avg shard | p99 | Block split |
|-------------------|-----------------|----------|--------------|------------|-------------|-----------|-----|-------------|
| 1× | 7,000 | 831 | 11,860 | 85.4% | 287 ns | 347 µs | 587 µs | 90.8% / 9.2% |
| 2× | 14,000 | 1,650 | 23,721 | 86.3% | 282 ns | 289 µs | **458 µs** | 48.7% / 51.3% |
| 4× | 28,000 | 3,330 | 47,443 | 86.4% | 249 ns | 293 µs | **446 µs** | 49.7% / 50.3% |
| 8× | 56,000 | 6,647 | 94,886 | 86.7% | 302 ns | 381 µs | 623 µs | 57.9% / 42.1% |
| 16× | 112,000 | 13,296 | 189,772 | 86.8% | 278 ns | 402 µs | 709 µs | 56.2% / 43.8% |
| 32× | 224,000 | 26,617 | 379,543 | 86.5% | 344 ns | 364 µs | 713 µs | 49.3% / 50.7% |

**1× is the only problematic case.** The 90.8%/9.2% block split at 7k workers is the lifetime skew issue — a few heavy branches landed on one shard and dominated it. At 2× the shards snap to balanced (48.7%/51.3%) due to the different hashes. and p99 drops to 458 µs.

**Sweet spot around 4×.** p99 is lowest at 4× (446 µs) where shards are balanced and per-shard tree depth is still moderate. Beyond that, p99 rises as each shard holds more blocks, but plateaus at ~700–750 µs between 16× and 32× even as workers double. This is sub-linear: block count grows 8× from 4× to 32×, but p99 grows only ~60%.

**No worker-count or lookup ceiling.** Throughput scales linearly (11,860 → 379,543 ops/s, doubling each factor step) with no saturation up to 224,000 effective workers. There is no cliff. Similarly, avg routing ranges from 249–344 ns across 831 to 26,617 branches — DashMap lookup is not a bottleneck in this range.

**Practical implication:** If minimizing single-query latency matters, 14k–28k workers is the sweet spot for a 2-shard config; if maximizing throughput matters, the indexer keeps scaling.

---

## Known Issues

### Lifetime block skew

`assign_shard` uses live block count as the primary load metric, so initial placement is good. The skew is a post-assignment problem: branches are placed permanently, and some branches accumulate far more blocks over their lifetime than others (e.g. long multi-turn conversations vs. one-shot queries). Shards diverge over time even though branch counts stay balanced.

Observed on `conversation_trace.jsonl`: 415 vs 416 branches (balanced) but 1,061,550 vs 104,300 blocks (10:1). The hot shard's larger tree drives up its p99.

**Future work — rebalancing:** periodically migrate heavy branches from the overloaded shard to lighter ones. This directly addresses lifetime skew when branches can be moved whole. It does not fully solve shared-prefix collapse (see `prefix_depth` section below), where the routing key itself is too coarse and a structural fix like node-depth routing is needed instead.

### `prefix_depth` must be tuned per workload

If most requests share a long system prompt, all conversations may hash to the same first `prefix_depth` blocks → single branch key → one shard gets all traffic. Set `prefix_depth` to span the shared prefix plus at least 1–2 unique blocks.
