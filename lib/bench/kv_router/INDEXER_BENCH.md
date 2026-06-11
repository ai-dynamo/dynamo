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

Use the Mooncake FAST25 arxiv trace as the primary benchmark trace unless you are intentionally testing a secondary workload. This is the trace most likely to match external benchmark discussions:

```bash
mkdir -p lib/kv-router/traces
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl \
  -o lib/kv-router/traces/mooncake_trace.jsonl
```

Secondary traces:

```bash
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl \
  -o lib/kv-router/traces/conversation_trace.jsonl
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl \
  -o lib/kv-router/traces/synthetic_trace.jsonl
curl -L https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl \
  -o lib/kv-router/traces/toolagent_trace.jsonl
```

| File | Description |
|------|-------------|
| `mooncake_trace.jsonl` | Primary arxiv workload: 23,608 requests over 1 hour, with two dominant hot prefixes |
| `conversation_trace.jsonl` | Smaller conversational workload: 12,031 requests, different hash sequences from the arxiv trace |
| `synthetic_trace.jsonl` | Synthetic workload: 3,993 requests, much smaller and structurally different |
| `toolagent_trace.jsonl` | Agentic/tool-use workload: same request count and similar distribution to the arxiv trace, but different hash sequences |

Trace shape summary from the current local copies:

| File | Requests | Span | Avg blocks/request | Unique depth-2 prefixes | Top depth-2 prefixes |
|------|---------:|------|-------------------:|------------------------:|----------------------|
| `mooncake_trace.jsonl` | 23,608 | 3,600,000 ms | 17.3 | 6,557 | `[46,47]` × 9,203; `[74,75]` × 3,449 |
| `conversation_trace.jsonl` | 12,031 | 3,536,999 ms | 24.0 | 7,373 | top prefix appears 43 times |
| `synthetic_trace.jsonl` | 3,993 | 1,022,025 ms | 30.5 | 1,138 | top prefixes appear 24 times |
| `toolagent_trace.jsonl` | 23,608 | 3,536,999 ms | 17.4 | 6,554 | `[46,47]` × 9,203; `[74,75]` × 3,449 |

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
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  concurrent-radix-tree-compressed --num-event-workers 8
```

**Branch-sharded depth=2 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

**Optional branch-sharded depth=4 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 4
```

### Peak throughput sweep

**CRTC baseline — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  concurrent-radix-tree-compressed --num-event-workers 8
```

**Branch-sharded depth=2 — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
```

### Worker scaling

```bash
for factor in 1 2 4 8 16 32; do
  cargo bench --package dynamo-bench --bench mooncake_bench -- \
    $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
    --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 \
    --num-unique-inference-workers 1000 \
    --trace-duplication-factor $factor \
    -d 7 \
    branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
done
```

### Repeated overload benchmark

This short-window benchmark intentionally drives offered load into the millions of request/event ops per second. The run is useful for stress-shape comparisons, but warnings are expected and the achieved values should not be interpreted as clean steady-state capacity.

**CRTC baseline (8 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --num-unique-inference-workers 128 \
  --trace-duplication-factor 20 \
  --trace-length-factor 4 \
  --benchmark-duration-ms 750 \
  --benchmark-runs 20 \
  concurrent-radix-tree-compressed \
  --num-event-workers 8
```

**Branch-sharded depth=2 with the same total event-worker count (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --num-unique-inference-workers 128 \
  --trace-duplication-factor 20 \
  --trace-length-factor 4 \
  --benchmark-duration-ms 750 \
  --benchmark-runs 20 \
  branch-sharded-crtc \
  --num-shards 2 \
  --num-event-workers-per-shard 4 \
  --prefix-depth 2
```

For a larger branch-sharded worker pool, use `--num-event-workers-per-shard 8` (16 total event workers with 2 shards).

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
| Dispatched | Queries routed to one shard for anchored suffix lookup |
| Shallow | Queries answered by router-owned TRIE state without shard dispatch |
| Avg routing | Routing-TRIE traversal time |
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
| `--shard-metrics-csv FILE` | — | Sample shard block/node counts over time → CSV |

### `branch-sharded-crtc` flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--num-shards` | 2 | Number of independent CRTC shards |
| `--num-event-workers-per-shard` | 4 | OS threads per shard for KV event processing |
| `--prefix-depth` | 2 | Maximum routing-trie depth before dispatching to one shard |

Note: `branch-sharded-crtc` uses trie/anchor routing and does not support approximate pruning. It provides a stronger routing-correctness model for out-of-order events, at the cost of higher routing latency and a hot-branch shard collapse risk on dominant-prefix workloads (see Known Issues).

---

## Results

Trace: `mooncake_trace.jsonl` (Mooncake FAST25 arxiv trace). Config: 2 shards × 4 workers for branch-sharded, 8 workers for CRTC baseline, `--trace-simulation-duration-ms 10000`, `--benchmark-duration-ms 30000`, `-d 7`.

### Steady-state — p99 at trace request rate (~17,268 ops/s offered)

| Indexer | Achieved ops/s | p99 | Routing outcome | Avg routing | Avg shard |
|---------|---------------|-----|-----------------|-------------|-----------|
| CRTC baseline (8w) | 17,201 | 1,093 µs | — | — | — |
| Branch-sharded depth=2 (2×4w) | 17,064 | 1,510 µs | 91.3% dispatched / 8.7% shallow | 388 µs | 161 µs |
| Branch-sharded depth=4 (2×4w) | 16,939 | 5,263 µs | 86.1% dispatched / 13.9% shallow | 678 µs | 244 µs |

All listed configs keep up with the offered trace rate. On this hot-prefix arxiv trace, branch-sharded routing is slower than CRTC in steady-state p99 because the routing TRIE does materially more work before dispatch and the stored block load collapses almost entirely onto one shard.

The true-miss rate remains effectively zero in these runs.

The routing TRIE provides a structural routing model that keeps shallow router state and shard anchors consistent, but it does not yet solve hot-branch load collapse. These numbers are useful for correctness and hot-prefix performance, but they do not show ideal multi-shard scaling.

> **Shard imbalance warning (this trace):** `mooncake_trace.jsonl` contains two dominant depth-2 prefixes: `[46,47]` appears 9,203 times and `[74,75]` appears 3,449 times. Branch-sharded routing can therefore collapse most stored blocks onto one shard until adaptive hot-branch splitting is implemented.

Shard block distribution:
```text
branch depth=2:  shard 0: 574 blocks (0.0%), 14 workers  shard 1: 2,094,155 blocks (100.0%), 7,000 workers
branch depth=4:  shard 0: 80,570 blocks (4.0%), 3,983 workers  shard 1: 1,922,501 blocks (96.0%), 7,000 workers
```

See Known Issues below for the current hot-branch collapse behavior.

### Peak throughput sweep — `mooncake_trace.jsonl`

Clean peak is defined as the highest achieved ops/s where the benchmark keeps up with the offered block throughput. Rows marked ⚠ were overloaded and should not be treated as clean throughput ceilings. The sweep runs shortest windows first for multithreaded indexers, so use the standalone steady-state run above for trace-rate p99.

| Indexer | Clean peak achieved ops/s | p99 at clean peak | Notes |
|---------|--------------------------:|-------------------|-------|
| CRTC baseline (8w) | **118,157** | 1,087 µs | Best clean throughput on this hot-prefix trace |
| Branch-sharded depth=2 (2×4w) | 16,699 | 12,332 µs | Sweep degrades heavily after overload; use steady-state run for normal p99 |

On this arxiv trace, CRTC has the highest clean throughput in the sweep. The hot-prefix distribution limits branch-sharded shard parallelism and caps clean peak throughput. This is the main reason the arxiv trace is a useful primary benchmark: it exposes the routing model under a less favorable and more realistic hot-prefix workload.

Full sweep data:

**CRTC baseline:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 30,000 ms | 17,268 | 17,219 | 1,027 µs |
| 18,455 ms | 28,071 | 27,955 | 741 µs |
| 11,352 ms | 45,635 | 45,309 | 771 µs |
| 6,983 ms | 74,187 | 73,298 | 683 µs |
| 4,296 ms | 120,589 | 118,157 | 1,087 µs |
| 2,643 ms ⚠ | 196,008 | 160,615 | 1,281 µs |
| 1,626 ms ⚠ | 318,603 | 178,812 | 1,236 µs |
| 1,000 ms ⚠ | 518,049 | 164,820 | 1,511 µs |

**Branch-sharded depth=2:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 30,000 ms | 17,268 | 16,699 | 12,332 µs |
| 18,455 ms ⚠ | 28,071 | 19,965 | 16,423 µs |
| 11,352 ms ⚠ | 45,635 | 23,088 | 14,922 µs |
| 6,983 ms ⚠ | 74,187 | 25,232 | 13,322 µs |
| 4,296 ms ⚠ | 120,589 | 26,185 | 11,384 µs |
| 2,643 ms ⚠ | 196,008 | 29,912 | 9,832 µs |
| 1,626 ms ⚠ | 318,603 | 35,720 | 6,970 µs |
| 1,000 ms ⚠ | 518,049 | 37,076 | 6,743 µs |

⚠ = bench warned it could not keep up with the offered rate.

### Worker scaling — branch-sharded depth=2

Config: 2 shards × 4 workers per shard, `--num-unique-inference-workers 1000`, `-d 7`, `--benchmark-duration-ms 30000`. `--trace-duplication-factor` duplicates request/hash spaces while keeping the worker identity count fixed; it increases branch diversity and event volume, but it does not multiply the number of worker identities.

| Trace duplication | Achieved / offered ops/s | p99 | Avg routing | Avg shard | Anchor installs / reuses | Block split |
|------------------:|--------------------------:|----:|------------:|----------:|-------------------------:|-------------|
| 1× | 17,015 / 17,268 | 3,034 µs | 455 µs | 203 µs | 88,228 / 0 | 0.0% / 100.0% |
| 2× | 33,554 / 34,536 | 6,383 µs | 515 µs | 297 µs | 176,386 / 14 | 91.9% / 8.1% |
| 4× ⚠ | 42,596 / 69,073 | 4,916 µs | 451 µs | 258 µs | 352,842 / 28 | 89.7% / 10.3% |
| 8× ⚠ | 40,695 / 138,147 | 5,489 µs | 473 µs | 280 µs | 705,593 / 98 | 76.6% / 23.4% |
| 16× ⚠ | 42,607 / 276,295 | 4,630 µs | 454 µs | 259 µs | 1,410,983 / 182 | 82.0% / 18.0% |
| 32× ⚠ | 35,018 / 552,590 | 5,928 µs | 529 µs | 319 µs | 2,822,050 / 343 | 84.8% / 15.2% |

1× and 2× keep up with the offered load. 4× and higher are overloaded with this 30s window and should be treated as stress-shape data rather than clean capacity.

Duplication increases branch diversity, but it does not reliably balance stored blocks because the hot routing path and branch lifetime skew still dominate. Achieved throughput saturates around 35-43k ops/s in the overloaded rows, and anchor installs scale roughly with event volume.

### Repeated overload benchmark

Config: `--num-unique-inference-workers 128`, `--trace-duplication-factor 20`, `--trace-length-factor 4`, `--benchmark-duration-ms 750`, `--benchmark-runs 20`. These runs generated 2,163,500 events: 1,007,958 `Stored` and 1,155,542 `Removed`. Every run warned that the benchmarker could not keep up, and macOS emitted malloc range-group warnings during event generation. Treat this as saturated stress data, not clean capacity.

| Indexer | Event workers | Offered ops/s | Achieved ops/s p50 | Achieved ops/s p90 | Achieved ops/s p99 | Achieved ops/s mean |
|---------|--------------:|--------------:|------------------:|------------------:|------------------:|-------------------:|
| CRTC baseline | 8 | 3,514,213 | 1,897,506 | 2,712,728 | 2,773,248 | 1,897,265 |
| Branch-sharded depth=2 | 2×4 | 3,514,213 | 594,934 | 664,920 | 703,730 | 593,921 |

| Indexer | Offered block ops/s | Achieved block ops/s p50 | Achieved block ops/s p90 | Achieved block ops/s p99 | Achieved block ops/s mean |
|---------|--------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|
| CRTC baseline | 73,600,216 | 39,740,576 | 56,814,248 | 58,081,760 | 39,735,535 |
| Branch-sharded depth=2 | 73,600,216 | 12,460,051 | 13,925,808 | 14,738,639 | 12,438,834 |

| Indexer | p99 latency p50 | p99 latency p90 | p99 latency p99 | p99 latency mean |
|---------|----------------:|----------------:|----------------:|-----------------:|
| CRTC baseline | 138 µs | 312 µs | 380 µs | 158 µs |
| Branch-sharded depth=2 | 352 µs | 484 µs | 560 µs | 370 µs |

Branch-sharded lands below CRTC on achieved ops/s in this overload run and has higher p99 lookup latency. Routing averages roughly 18-27 µs, shard traversal roughly 12-19 µs, and stored blocks still collapse heavily onto one shard (usually about 93-100% on shard 1), so this is not a balanced-sharding result. This command is dominated by event processing, especially `Removed` events; branch-sharded adds routing-TRIE and anchor bookkeeping on that path while CRTC applies events directly.

---

## Known Issues

### Hot-branch shard collapse and lifetime block skew

`BranchShardedIndexer` uses a routing TRIE with static divergent-child shard assignment: the first child under a routing node stays with the parent shard, and later divergent siblings may split to other shards. Two skew modes remain:

1. **Shared-prefix collapse:** if many requests share the same first `prefix_depth` blocks, they share the same routing path and land on one shard. On `mooncake_trace.jsonl`, the two hottest depth-2 prefixes account for 12,652 of 23,608 requests, and the branch-sharded depth=2 steady-state run above stores effectively all blocks on one shard.
2. **Lifetime skew:** even when branch counts are distributed, branches are placed permanently and some conversations accumulate far more blocks than others. Over time, a few heavy branches can dominate one shard.

The hot shard's larger tree drives up p99.

**Future work — hot-branch splitting/rebalancing:** detect prefixes whose request or pending-prefill-token load would overload a shard, then split or migrate that hot branch across a small candidate shard set. This directly addresses the current collapse mode while preserving single-shard routing for cold branches.

### `prefix_depth` must be tuned per workload

If most requests share a long prompt prefix, all conversations may follow the same routing-TRIE path until the first divergent block. Set `prefix_depth` to span the shared prefix plus at least 1-2 unique blocks when you want the configured depth to expose more branch diversity.

The routing TRIE avoids prefix-key collisions, but `prefix_depth` tuning alone does not solve a genuinely dominant hot branch. The code has an open TODO for adaptive hot-branch splitting. Until that is resolved, compare branch-sharded results on traces with narrow prefix diversity before choosing it for production.
