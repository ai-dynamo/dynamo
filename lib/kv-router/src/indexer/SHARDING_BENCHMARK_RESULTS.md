<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sharding Benchmark Results

This document summarizes benchmark comparisons for four KV-router indexer modes:

- `concurrent-radix-tree-compressed` (CRTC baseline, no sharding)
- Option A: `branch-sharded-crtc`
- Option B: `prefix-sharded-crtc`
- Option C: `virtual-shard-sharded-crtc`

The goal was to compare:

- current least-loaded / stateful branch placement
- fully deterministic direct routing
- deterministic logical routing with stateful ownership and ownership migration

## Benchmark Setup

Build/runner:

```bash
cargo bench --package dynamo-bench --bench mooncake_bench --no-run
target/release/deps/mooncake_bench-<hash> ...
```

Steady-state config:

- `--trace-simulation-duration-ms 10000`
- `--benchmark-duration-ms 30000`
- `-d 7`
- baseline: `--num-event-workers 8`
- sharded runs: `--num-shards 2 --num-event-workers-per-shard 4`
- Option C: `--num-virtual-shards 8`
- prefix depths tested: `1`, `2`, and `4`

Traces:

- `conversation_trace.jsonl`
- `synthetic_trace.jsonl`
- `toolagent_trace.jsonl`

## Caveats

- Option C currently has benchmark-oriented rebalance defaults hardcoded in [mooncake_shared.rs](/Users/hannahz/.codex/worktrees/fb4c/dynamo/lib/bench/kv_router/mooncake_shared.rs):
  - rebalance interval: `2s`
  - imbalance threshold: `1.4`
  - dual-write window: `1s`
- Those are not yet exposed as CLI flags, so the current results should be read as one reasonable migrated-C configuration, not as a fully tuned optimum.
- The public `synthetic_trace.jsonl` did not work through the current replay pipeline with the default worker partitioning. To run it successfully, the benchmark used:
  - `--num-unique-inference-workers 1`
- Option C is now a real migrated-ownership prototype, but it is still a benchmark implementation rather than a finalized production control-plane design.

## Trace Characteristics

### `conversation_trace.jsonl`

- 12,031 requests
- every request starts with block `0`
- this is effectively a universal system-prompt workload
- `K=1` routing is unusable here because all traffic collapses onto one shard (`CV ~= 2.0`)
- `K=2` becomes nearly perfectly balanced by request count (`CV ~= 0.005`)
- beyond the first block, reuse is actually fairly low
- multi-turn rate is negligible

What it stresses:

- single shared-prefix collapse
- sensitivity to `prefix_depth`

### `synthetic_trace.jsonl`

- 3,993 requests
- 2,211 distinct first blocks
- no universal prefix
- naturally balanced even at `K=1` (`CV ~= 0.024`)
- highest reuse ratio of the three traces
- some groups share very long prefixes before diverging

What it stresses:

- long shared depth within already-separated groups
- shallow-query and block-count behavior
- less useful for separating routing strategies because many approaches perform similarly

### `toolagent_trace.jsonl`

- 23,608 requests
- dominated by three first-block clusters:
  - block `0` (~46%)
  - block `46` (~39%)
  - block `74` (~15%)
- one cluster has a long shared prefix (~12 blocks / ~6k tokens)
- deterministic prefix routing remains skewed even for deeper `K` (`K=2 CV ~= 0.74`)

What it stresses:

- multi-agent / multi-cluster prompt structure
- the limit of prefix-based sharding
- where node-depth routing is most likely to matter

### Short Version

| Trace | Character | K=1 balance | K=2 balance | Main challenge |
|---|---|---|---|---|
| `conversation_trace` | universal system prompt | broken | excellent | must use `K >= 2` |
| `synthetic_trace` | diverse, no shared prefix | good | good | not very discriminating |
| `toolagent_trace` | three-cluster multi-agent | broken | still skewed | prefix sharding may be fundamentally insufficient |

## Steady-State Results

### Conversation Trace

| Variant | Achieved ops/s | p99 us | Early-exit | Migrations | Block split |
|---|---:|---:|---:|---:|---|
| CRTC baseline | 11608 | 5710 | — | 0 | — |
| A branch-sharded d=1 | 11621 | 2822 | 0.0% | 0 | 100.0% / 0.0% |
| A branch-sharded d=2 | 11751 | 1347 | 85.4% | 0 | 8.3% / 91.7% |
| A branch-sharded d=4 | 11785 | 1270 | 87.0% | 0 | 91.1% / 8.9% |
| B prefix-sharded d=1 | 11441 | 7831 | — | 0 | 100.0% / 0.0% |
| B prefix-sharded d=2 | 11774 | 1631 | — | 0 | 9.6% / 90.4% |
| B prefix-sharded d=4 | 11776 | 1499 | — | 0 | 91.0% / 9.0% |
| C virtual-shard d=1 | 11624 | 5802 | 0.0% | 2 | 68.7% / 31.3% |
| C virtual-shard d=2 | 11767 | 1627 | 85.4% | 2 | 31.4% / 68.6% |
| C virtual-shard d=4 | 11764 | 1426 | 87.0% | 2 | 31.5% / 68.5% |

Summary:

- A is still the best steady-state choice on this trace.
- `prefix_depth=1` is effectively unusable on this workload for direct prefix routing: A and B collapse onto one physical shard, and C only partially recovers by migrating ownership after the fact.
- C materially improves block balance relative to A/B.
- C did not beat A on p99 here, even after ownership migration.
- All sharded variants are much better than the unsharded baseline.

### Synthetic Trace

Note: these runs used `--num-unique-inference-workers 1` due to current replay-pipeline assumptions.

| Variant | Achieved ops/s | p99 us | Early-exit | Migrations | Block split |
|---|---:|---:|---:|---:|---|
| CRTC baseline | 4888 | 15 | — | 0 | — |
| A branch-sharded d=1 | 4888 | 5 | 86.7% | 0 | 51.1% / 48.9% |
| A branch-sharded d=2 | 4888 | 10 | 75.6% | 0 | 50.0% / 50.0% |
| A branch-sharded d=4 | 4888 | 10 | 86.1% | 0 | 48.6% / 51.4% |
| B prefix-sharded d=1 | 4888 | 7 | — | 0 | 28.7% / 71.3% |
| B prefix-sharded d=2 | 4888 | 12 | — | 0 | 29.1% / 70.9% |
| B prefix-sharded d=4 | 4888 | 19 | — | 0 | 26.1% / 73.9% |
| C virtual-shard d=1 | 4887 | 8 | 76.1% | 0 | 51.2% / 48.8% |
| C virtual-shard d=2 | 4888 | 6 | 86.4% | 0 | 53.2% / 46.8% |
| C virtual-shard d=4 | 4888 | 11 | 76.0% | 0 | 46.6% / 53.4% |

Summary:

- Throughput is identical across all variants.
- `prefix_depth=1` is fine on this trace because the first block is already diverse; this is the least discriminating workload of the three.
- C d=2 produced the best p99 in this setup.
- B is the weakest on both latency and balance.
- This trace does not separate the options as strongly as the other two.

### Toolagent Trace

| Variant | Achieved ops/s | p99 us | Early-exit | Migrations | Block split |
|---|---:|---:|---:|---:|---|
| CRTC baseline | 17026 | 3661 | — | 0 | — |
| A branch-sharded d=1 | 17016 | 2123 | 0.0% | 0 | 90.9% / 9.1% |
| A branch-sharded d=2 | 16876 | 1731 | 39.5% | 0 | 31.0% / 69.0% |
| A branch-sharded d=4 | 17091 | 2216 | 40.1% | 0 | 32.0% / 68.0% |
| B prefix-sharded d=1 | 17006 | 3711 | — | 0 | 100.0% / 0.0% |
| B prefix-sharded d=2 | 17106 | 1901 | — | 0 | 60.2% / 39.8% |
| B prefix-sharded d=4 | 16820 | 4708 | — | 0 | 70.7% / 29.3% |
| C virtual-shard d=1 | 17003 | 5843 | 0.0% | 0 | 64.6% / 35.4% |
| C virtual-shard d=2 | 17081 | 1447 | 39.5% | 0 | 58.8% / 41.2% |
| C virtual-shard d=4 | 17059 | 2469 | 40.1% | 0 | 68.7% / 31.3% |

Summary:

- C d=2 is the best steady-state result on this trace.
- `prefix_depth=1` is again too coarse here: B fully collapses, A is badly skewed, and C cannot make up the difference without a better logical routing key.
- A d=2 is also strong.
- B d=4 regresses badly.
- This is the most revealing workload for distinguishing routing strategies.

## Conversation Peak-Throughput Sweep

Representative sweep on `conversation_trace.jsonl`:

- baseline
- A d=4
- B d=4
- C d=4

| Variant | Peak achieved ops/s | p99 at peak (us) | Best achieved block ops/s |
|---|---:|---:|---:|
| CRTC baseline | 40288 | 6186 | 469616 |
| A branch-sharded d=4 | 247716 | 773 | 2887444 |
| B prefix-sharded d=4 | 46657 | 7395 | 543849 |
| C virtual-shard d=4 | 255316 | 731 | 2976035 |

Summary:

- A and C both massively outperform the unsharded baseline under saturation.
- C d=4 slightly outperformed A d=4 in this sweep.
- B remains much closer to baseline than to A/C under heavy load.

## Overall Takeaways

- Option A remains the strongest all-around steady-state choice.
- `prefix_depth=1` behaves exactly like the trace characterization predicts: it is acceptable on `synthetic_trace`, but clearly too coarse on `conversation_trace` and `toolagent_trace`.
- Option B is the simplest, but it is also the most sensitive to routing-key quality.
- Option C is now a genuine migrated-ownership prototype rather than just an indirection wrapper.
- Option C's current numbers should be interpreted with its benchmark-only hardcoded rebalance defaults in mind.
- Option C looks especially promising on:
  - `toolagent_trace` steady state
  - `synthetic_trace` steady state
  - `conversation_trace` peak throughput
- `toolagent_trace` remains the clearest motivation for exploring node-depth routing next.

## Raw Outputs

Raw benchmark logs were captured locally under:

- `/tmp/dep-bench/matrix`
- `/tmp/dep-bench/sweep`
