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

**Branch-sharded depth=4 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 4
```

**Anchor-aware branch-sharded depth=2 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 -d 7 \
  anchor-aware-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
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

**Anchor-aware branch-sharded depth=2 — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 1000 --sweep-max-ms 30000 --sweep-steps 8 \
  anchor-aware-branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 2
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

**Anchor-aware branch-sharded depth=2 with the same total event-worker count (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --num-unique-inference-workers 128 \
  --trace-duplication-factor 20 \
  --trace-length-factor 4 \
  --benchmark-duration-ms 750 \
  --benchmark-runs 20 \
  anchor-aware-branch-sharded-crtc \
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
| Dispatched | Queries routed to one shard for lookup |
| Early-exit % | Queries with no registered prefix alias, resolved without shard dispatch |
| Avg routing | Prefix-key routing-table lookup time |
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

Trace: `mooncake_trace.jsonl` (Mooncake FAST25 arxiv trace). Config: 2 shards × 4 workers for branch-sharded, 8 workers for CRTC baseline, `--trace-simulation-duration-ms 10000`, `--benchmark-duration-ms 30000`, `-d 7`.

### Steady-state — p99 at trace request rate (~17,268 ops/s offered)

| Indexer | Achieved ops/s | p99 | Routing outcome | Avg routing | Avg shard |
|---------|---------------|-----|-----------------|-------------|-----------|
| CRTC baseline (8w) | 17,201 | 1,093 µs | — | — | — |
| Branch-sharded depth=2 (2×4w) | 17,182 | **933 µs** | 0.0% miss | 762 ns | 238 µs |
| Branch-sharded depth=4 (2×4w) | 17,221 | **841 µs** | 0.0% miss | 717 ns | 237 µs |
| Anchor-aware BSI depth=2 (2×4w) | 17,064 | 1,510 µs | 91.3% dispatched / 8.7% shallow | 388 µs | 161 µs |

All indexers keep up with the offered trace rate. On this arxiv trace, branch-sharded depth=2 is about **15% lower p99** than CRTC, and depth=4 is about **23% lower p99**. This is a modest latency win, not the large win seen on the smaller `conversation_trace.jsonl` workload; do not compare those results directly.

The true-miss rate remains effectively zero in these runs.

Anchor-aware BSI is slower than both CRTC and branch-sharded in this steady-state run. Its routing TRIE provides a stronger structural routing model, but on this hot-prefix trace the routing work is materially higher and the shard load collapses almost entirely onto one shard.

> **Shard imbalance warning (this trace):** `mooncake_trace.jsonl` contains two dominant depth-2 prefixes: `[46,47]` appears 9,203 times and `[74,75]` appears 3,449 times. Branch-sharded routing therefore remains skewed even though it does not collapse to a single shard. These numbers are useful for correctness and hot-prefix performance, but they do not show ideal multi-shard scaling.

Shard block distribution:
```text
branch depth=2:  shard 0: 1,919,603 blocks (87.0%), 7,000 workers  shard 1: 286,468 blocks (13.0%), 7,000 workers
                 branches: shard[0]=819, shard[1]=3

branch depth=4:  shard 0: 1,919,603 blocks (87.0%), 7,000 workers  shard 1: 286,468 blocks (13.0%), 7,000 workers
                 branches: shard[0]=960, shard[1]=17

anchor-aware:    shard 0: 574 blocks (0.0%), 14 workers  shard 1: 2,094,155 blocks (100.0%), 7,000 workers
```

Increasing `prefix_depth` from 2 to 4 creates more branch aliases, but the stored block split is unchanged because the hottest branch lifetime still dominates the trace. See Known Issues below.

### Peak throughput sweep — `mooncake_trace.jsonl`

Clean peak is defined as the highest achieved ops/s where the benchmark keeps up with the offered block throughput. Rows marked ⚠ were overloaded and should not be treated as clean throughput ceilings. The sweep runs shortest windows first for multithreaded indexers, so use the standalone steady-state run above for trace-rate p99.

| Indexer | Clean peak achieved ops/s | p99 at clean peak | Notes |
|---------|--------------------------:|-------------------|-------|
| CRTC baseline (8w) | **118,157** | 1,087 µs | Best clean throughput on this hot-prefix trace |
| Branch-sharded depth=2 (2×4w) | 73,304 | 1,337 µs | Clean peak is lower than CRTC because traffic is skewed to one shard |
| Anchor-aware BSI depth=2 (2×4w) | 16,699 | 12,332 µs | Sweep degrades heavily after overload; use steady-state run for normal p99 |

On this arxiv trace, CRTC has the highest clean throughput in the sweep. That is not a contradiction with the steady-state table: branch-sharded still has lower trace-rate p99, but the hot-prefix distribution limits shard parallelism and therefore caps clean peak throughput. This is the main reason the arxiv trace is a better primary benchmark: it exposes the accuracy fix under a less favorable and more realistic hot-prefix workload.

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
| 30,000 ms ⚠ | 17,268 | 13,732 | 17,772 µs |
| 18,455 ms | 28,071 | 27,967 | 1,967 µs |
| 11,352 ms | 45,635 | 45,089 | 814 µs |
| 6,983 ms | 74,187 | 73,304 | 1,337 µs |
| 4,296 ms ⚠ | 120,589 | 73,655 | 6,186 µs |
| 2,643 ms ⚠ | 196,008 | 113,628 | 2,663 µs |
| 1,626 ms ⚠ | 318,603 | 156,775 | 1,383 µs |
| 1,000 ms ⚠ | 518,049 | 117,445 | 2,390 µs |

**Anchor-aware BSI depth=2:**

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

Tests how p99 and shard balance change as trace volume grows. Config: 2 shards × 4 workers per shard, `--num-unique-inference-workers 1000`, `-d 7` (7 replicas × 1,000 workers = 7,000 worker identities), `--benchmark-duration-ms 30000`.

Note: `--trace-duplication-factor` duplicates request/hash spaces while keeping the worker identity count fixed at 7,000. It increases branch diversity and event volume; it does not multiply the number of worker identities.

| Trace duplication | Branches | Offered ops/s | Achieved ops/s | Avg routing | Avg shard | p99 | Block split |
|------------------:|---------:|--------------:|---------------:|-------------|-----------|-----|-------------|
| 1× | 822 | 17,268 | 16,942 | 4.3 µs | 419 µs | 9,069 µs | 87.1% / 12.9% |
| 2× | 1,640 | 34,536 | 33,623 | 4.3 µs | 531 µs | 9,989 µs | 45.5% / 54.5% |
| 4× ⚠ | 3,306 | 69,073 | 56,049 | 5.2 µs | 488 µs | 8,403 µs | 54.2% / 45.8% |
| 8× ⚠ | 6,587 | 138,147 | 61,306 | 4.8 µs | 439 µs | 6,563 µs | 29.4% / 70.6% |
| 16× ⚠ | 13,162 | 276,295 | 106,127 | 2.0 µs | 262 µs | 2,859 µs | 44.3% / 55.7% |
| 32× ⚠ | 26,328 | 552,590 | 99,795 | 4.9 µs | 263 µs | 2,549 µs | 55.4% / 44.6% |

**1× and 2× keep up but p99 is noisy in this long stress sequence.** Use the standalone steady-state runs for normal trace-rate latency comparisons.

**Duplication improves branch diversity.** The block split is much closer to balanced at 2×, 4×, 16×, and 32× than in the single-copy arxiv run. The 8× run still skews because branch lifetimes are uneven even when branch counts grow.

**4× and higher are overloaded with this 30s window on this machine.** Those rows are useful as stress shape, not as clean throughput limits. The benchmark driver emits warnings once achieved block throughput falls behind offered block throughput.

**Practical implication:** for this arxiv trace and a 2-shard config, branch diversity alone does not guarantee clean scaling. The hot-prefix distribution and lifetime skew still matter, and higher-pressure runs need either more shards, longer benchmark windows, or lower trace volume to separate indexer limits from benchmark-driver pressure.

### Repeated overload benchmark

Config: `--num-unique-inference-workers 128`, `--trace-duplication-factor 20`, `--trace-length-factor 4`, `--benchmark-duration-ms 750`, `--benchmark-runs 20`. These runs generated 2,163,500 events: 1,007,958 `Stored` and 1,155,542 `Removed`. Every run warned that the benchmarker could not keep up, and macOS emitted malloc range-group warnings during event generation. Treat this as saturated stress data, not clean capacity.

| Indexer | Event workers | Offered ops/s | Achieved ops/s p50 | Achieved ops/s p90 | Achieved ops/s p99 | Achieved ops/s mean |
|---------|--------------:|--------------:|------------------:|------------------:|------------------:|-------------------:|
| CRTC baseline | 8 | 3,514,213 | 1,897,506 | 2,712,728 | 2,773,248 | 1,897,265 |
| Branch-sharded depth=2 | 2×4 | 3,514,213 | 423,942 | 451,508 | 457,397 | 411,244 |
| Anchor-aware BSI depth=2 | 2×4 | 3,514,213 | 594,934 | 664,920 | 703,730 | 593,921 |

| Indexer | Offered block ops/s | Achieved block ops/s p50 | Achieved block ops/s p90 | Achieved block ops/s p99 | Achieved block ops/s mean |
|---------|--------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|
| CRTC baseline | 73,600,216 | 39,740,576 | 56,814,248 | 58,081,760 | 39,735,535 |
| Branch-sharded depth=2 | 73,600,216 | 8,878,863 | 9,456,199 | 9,579,526 | 8,612,927 |
| Anchor-aware BSI depth=2 | 73,600,216 | 12,460,051 | 13,925,808 | 14,738,639 | 12,438,834 |

| Indexer | p99 latency p50 | p99 latency p90 | p99 latency p99 | p99 latency mean |
|---------|----------------:|----------------:|----------------:|-----------------:|
| CRTC baseline | 138 µs | 312 µs | 380 µs | 158 µs |
| Branch-sharded depth=2 | 137 µs | 161 µs | 198 µs | 137 µs |
| Anchor-aware BSI depth=2 | 352 µs | 484 µs | 560 µs | 370 µs |

Branch-sharded lookup latency remains comparable to CRTC in this overload run: routing averages roughly 1.2-2.6 µs, shard traversal averages roughly 9-16 µs, and block placement stays reasonably balanced across the two shards. The lower achieved ops/s is therefore not a lookup-latency regression. This command is dominated by event processing, especially `Removed` events; branch-sharded adds routing-table and block-to-shard bookkeeping on that path while CRTC applies events directly.

Anchor-aware BSI lands between CRTC and branch-sharded on achieved ops/s in this overload run, but with higher p99 lookup latency. Routing averages roughly 18-27 µs, shard traversal roughly 12-19 µs, and stored blocks still collapse heavily onto one shard (usually about 93-100% on shard 1), so this is not a balanced-sharding result.

---

## Known Issues

### Shared-prefix shard collapse and lifetime block skew

`assign_shard` uses live block count as the primary load metric, so branch placement adapts to observed load. Two skew modes remain:

1. **Shared-prefix collapse:** if many requests share the same first `prefix_depth` blocks, they share the same routing aliases and land on one shard. On `mooncake_trace.jsonl`, the two hottest depth-2 prefixes account for 12,652 of 23,608 requests, and branch-sharded depth=2/depth=4 store 87% of blocks on one shard.
2. **Lifetime skew:** even when branch counts are distributed, branches are placed permanently and some conversations accumulate far more blocks than others. Over time, a few heavy branches can dominate one shard.

The hot shard's larger tree drives up p99.

**Future work — rebalancing:** periodically migrate heavy branches from the overloaded shard to lighter ones. This directly addresses lifetime skew when branches can be moved whole. It does not fully solve shared-prefix collapse (see `prefix_depth` section below), where the routing key itself is too coarse and a structural fix like node-depth routing is needed instead.

### `prefix_depth` must be tuned per workload

If most requests share a long prompt prefix, all conversations may hash to the same first `prefix_depth` blocks → single branch key → one shard gets most traffic. Set `prefix_depth` to span the shared prefix plus at least 1–2 unique blocks.

Note: `anchor-aware-branch-sharded-crtc` avoids FNV routing-key collisions (different conversations with the same prefix still get distinct TRIE paths) but has its own hot-branch collapse issue on dominant-prefix workloads (see below). Neither variant is unconditionally better here — tune `prefix_depth` regardless of which you use.

### Anchor-aware BSI: hot-branch shard collapse

`AnchorAwareBranchShardedIndexer` uses static divergent-shard assignment: the first conversation under a new parent node stays on the parent's shard; only subsequent divergent siblings are hashed to a different shard. On workloads with a dominant shared prefix, nearly all traffic can end up as the "first child" of the same TRIE node and route to one shard. Observed on `mooncake_trace.jsonl`: effectively 100% of stored blocks on shard 1 in the steady-state run.

The code has an open TODO for adaptive hot-branch splitting. Until that is resolved, compare both branch-sharded variants on traces with narrow prefix diversity before choosing one for production.
