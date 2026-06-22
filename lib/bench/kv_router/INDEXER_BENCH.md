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

**Branch-sharded depth=3 (2 shards × 4 workers):**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 --benchmark-duration-ms 30000 --benchmark-runs 5 -d 7 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 3
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
  --sweep --sweep-min-ms 750 --sweep-max-ms 60000 --sweep-steps 10 \
  concurrent-radix-tree-compressed --num-event-workers 8
```

**Branch-sharded depth=3 — sweep:**
```bash
cargo bench --package dynamo-bench --bench mooncake_bench -- \
  $(git rev-parse --show-toplevel)/lib/kv-router/traces/mooncake_trace.jsonl \
  --trace-simulation-duration-ms 10000 -d 7 \
  --sweep --sweep-min-ms 750 --sweep-max-ms 60000 --sweep-steps 10 \
  branch-sharded-crtc --num-shards 2 --num-event-workers-per-shard 4 --prefix-depth 3
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

Note: `branch-sharded-crtc` uses trie/anchor routing and does not support approximate pruning. It provides a stronger routing-correctness model for out-of-order events, at the cost of higher routing latency and residual hot-continuation or lifetime-skew risks on dominant-prefix workloads (see Known Issues).

---

## Results

Trace: `mooncake_trace.jsonl` (Mooncake FAST25 arxiv trace). Config: 2 shards × 4 workers for branch-sharded depth=3, 8 workers for CRTC baseline, `--trace-simulation-duration-ms 10000`, `--benchmark-duration-ms 30000`, `--benchmark-runs 5`, `-d 7`.

For interpretation of the BSI correctness/performance tradeoff and next-step recommendations, see [`BSI_FINDINGS.md`](./BSI_FINDINGS.md).

### Steady-state — p99 at trace request rate (~17,268 ops/s offered)

| Indexer | Achieved ops/s mean | Block ops/s mean | p99 mean | p99 p50 / p90 / max | Routing outcome | Avg routing | Avg shard | Block split |
|---------|--------------------:|-----------------:|---------:|---------------------:|-----------------|------------:|----------:|-------------|
| CRTC baseline (8w) | 17,217.3 | 168,552.7 | 1,538 µs | 1,494 / 1,924 / 1,924 µs | — | — | — | — |
| Branch-sharded depth=3 (2×4w) | 16,806.3 | 164,528.8 | 4,830 µs | 4,863 / 6,672 / 6,672 µs | 88.1% dispatched / 11.9% shallow | 393 µs | 327 µs | 55.5% / 44.5% |

Both configs keep up with the offered trace rate. Corrected deterministic child sharding removes the previous all-block shard collapse, but branch-sharded still trails CRTC by about 2.4% on achieved ops/s and has about 3.1x higher mean p99 on this 5-run steady-state comparison.

The true-miss rate remains effectively zero in these runs.

The routing TRIE provides a structural routing model that keeps shallow router state and shard anchors consistent. It now spreads divergent children by sequence hash instead of preserving creation-order placement, but it still does not prove ideal multi-shard scaling for every hot-prefix workload.

> **Shard imbalance warning (this trace):** `mooncake_trace.jsonl` contains two dominant depth-2 prefixes: `[46,47]` appears 9,203 times and `[74,75]` appears 3,449 times. Deterministic child sharding spreads their observed continuations in the 2-shard runs above, but a genuinely dominant single continuation can still overload one shard until adaptive hot-branch splitting is implemented.

Shard block distribution:
```text
branch depth=3:  shard 0: 1,161,601 blocks (55.5%), 7,000 workers  shard 1: 930,496 blocks (44.5%), 6,951 workers
```

See Known Issues below for the remaining hot-continuation and lifetime-skew behavior.

### 3x pressure — 10s benchmark window (~51,805 ops/s offered)

| Indexer | Achieved ops/s mean | Block ops/s mean | p99 mean | Warning? | Avg routing | Avg shard |
|---------|--------------------:|-----------------:|---------:|----------|------------:|----------:|
| CRTC baseline (8w) | 51,377.6 | 502,973.4 | 1,339 µs | No | — | — |
| Branch-sharded depth=3 (2×4w) | 38,769.1 | 379,539.5 | 6,853 µs | Yes | 420 µs | 409 µs |

At 3x pressure, CRTC still keeps up with offered load while branch-sharded saturates. This is the clearest local evidence that the remaining branch-sharded bottleneck is not shard placement alone: the block split is stable at about 55.5% / 44.5%, but exact router scoring plus shard dispatch/merge cost dominates under pressure.

### Peak throughput and ceiling — `mooncake_trace.jsonl`

Clean peak is defined here as the highest targeted run where achieved ops/s stayed at or above 95% of offered ops/s and the benchmark did not warn. The raw sweep is useful for finding candidate windows, but several sweep rows were noisier than standalone reruns in this pass, so targeted reruns are the source of truth for capacity claims.

| Indexer | Highest clean targeted offered ops/s | Achieved ops/s mean | p99 mean | First unstable/overloaded target | Notes |
|---------|-----------------------------------:|--------------------:|---------:|----------------------------------:|-------|
| CRTC baseline (8w) | 148,014.0 | **143,174.7** | 896 µs | 160,287.4 offered ops/s, 151,197.9 achieved, below 95% keep-up | About 3.5x branch-sharded clean throughput by achieved ops/s |
| Branch-sharded depth=3 (2×4w) | 43,170.8 | 41,429.0 | 3,520 µs | 45,047.7 offered ops/s; two reruns warned | Correct and balanced, but exact routing/merge work caps clean throughput |

On this arxiv trace, CRTC has the highest clean throughput. Branch-sharded reaches higher achieved rates in overloaded rows, but those runs warn or fail the 95% keep-up threshold and should not be used as clean capacity claims.

Targeted rerun data:

**CRTC baseline:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 7,000 ms | 74,007.0 | 73,219.3 | 648 µs |
| 6,000 ms | 86,341.5 | 84,124.1 | 814 µs |
| 5,259 ms | 98,507.1 | 97,260.7 | 702 µs |
| 4,000 ms | 129,512.2 | 126,750.0 | 1,078 µs |
| 3,500 ms | 148,014.0 | 143,174.7 | 896 µs |
| 3,232 ms ⚠ | 160,287.4 | 151,197.9 | 1,164 µs |

**Branch-sharded depth=3:**

| Benchmark window | Offered ops/s | Achieved ops/s | p99 |
|-----------------|--------------|----------------|-----|
| 22,659 ms | 22,862.8 | 22,406.3 | 2,020 µs |
| 13,925 ms | 37,202.8 | 36,043.0 | 2,641 µs |
| 12,000 ms | 43,170.8 | 41,429.0 | 3,520 µs |
| 11,500 ms ⚠ | 45,047.7 | 40,849.8 | 5,202 µs |
| 11,500 ms rerun ⚠ | 45,047.7 | 38,574.2 | 6,190 µs |
| 11,000 ms ⚠ | 47,095.4 | 43,373.4 | 4,446 µs |
| 10,000 ms ⚠ | 51,804.9 | 38,769.1 | 6,853 µs |

⚠ = overloaded by warning or by falling below the 95% keep-up threshold.

### Duplicated-Mooncake stress

Config: `--trace-duplication-factor 2`, `--benchmark-duration-ms 30000`, `--benchmark-runs 5`, same CRTC and BSI worker counts as above.

| Source | Indexer | Warnings | Achieved ops/s mean | Block ops/s mean | p99 mean |
|--------|---------|---------:|--------------------:|-----------------:|---------:|
| Earlier stress run | CRTC baseline (8w) | 0 | 34,379 | — | 6,238 µs |
| Earlier stress run | BSI before borrowed suffixes | 0 | 33,758 | — | 5,822 µs |
| Earlier stress run | BSI with borrowed suffixes | 0 | 34,216 | — | 2,072 µs |
| Current rerun | CRTC baseline (8w) | 0 | 34,447.0 | 337,220.6 | 1,094 µs |
| Current rerun | Branch-sharded depth=3 (2×4w) | 0 | 34,225.3 | 335,050.4 | 2,215 µs |

The current BSI rerun sustains the duplicated target and stays near a 2 ms p99. The current same-run CRTC baseline is stronger, so use the current same-run comparison for present-day BSI-vs-CRTC claims.

### Historical worker scaling — branch-sharded depth=2

This section is retained as older stress-shape evidence. It was not rerun in the current pass after deterministic child placement and borrowed suffixes, so do not use it as a current BSI-vs-CRTC capacity claim.

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

### Historical repeated overload benchmark

This saturated stress run is also older and was not rerun after the current BSI changes. It remains useful for command shape and overload behavior, not for current capacity claims.

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

Branch-sharded landed below CRTC on achieved ops/s in this overload run and had higher p99 lookup latency. Routing averaged roughly 18-27 µs, shard traversal roughly 12-19 µs, and stored blocks collapsed heavily onto one shard (usually about 93-100% on shard 1), so this was not a balanced-sharding result. This command was dominated by event processing, especially `Removed` events; branch-sharded added routing-TRIE and anchor bookkeeping on that path while CRTC applied events directly.

---

## Known Issues

### Hot-continuation shard collapse and lifetime block skew

`BranchShardedIndexer` uses a routing TRIE with deterministic divergent-child shard assignment based on each child's sequence hash. The old creation-order behavior, where the first child under a routing node stayed with the parent shard, is no longer used. Two skew modes remain:

1. **Hot-continuation collapse:** if many requests share the same first block after `prefix_depth`, they still share the same routed child and land on one shard. On `mooncake_trace.jsonl`, the corrected two-shard steady-state runs above are balanced, but this is workload-dependent.
2. **Lifetime skew:** even when branch counts are distributed, branches are placed permanently and some conversations accumulate far more blocks than others. Over time, a few heavy branches can dominate one shard.

The hot shard's larger tree drives up p99.

**Future work — hot-branch splitting/rebalancing:** detect prefixes whose request or pending-prefill-token load would overload a shard, then split or migrate that hot branch across a small candidate shard set. This directly addresses the current collapse mode while preserving single-shard routing for cold branches.

### `prefix_depth` must be tuned per workload

If most requests share a long prompt prefix, all conversations may follow the same routing-TRIE path until the first divergent block. Set `prefix_depth` to span the shared prefix plus at least 1-2 unique blocks when you want the configured depth to expose more branch diversity.

The routing TRIE avoids prefix-key collisions, but `prefix_depth` tuning alone does not solve a genuinely dominant hot branch. The code has an open TODO for adaptive hot-branch splitting. Until that is resolved, compare branch-sharded results on traces with narrow prefix diversity before choosing it for production.
