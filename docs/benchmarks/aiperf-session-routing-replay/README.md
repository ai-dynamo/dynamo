---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AIPerf Session Routing Replay
subtitle: Reproduce round-robin, sticky-session, KV-router, and sticky-proxy replay comparisons
---

# AIPerf Session Routing Replay

## Simulation Results

These simulation results use the first `10,000` rows of
`/Users/peabrane/Downloads/dataset_aiperf.jsonl`, interpreted as per-session deltas with
`--trace-format mooncake_delta`.

Config: `8` workers, closed-loop concurrency `128`, trace block size `64`, engine block size `64`,
`16,384` GPU KV blocks per worker, AIC-backed vLLM timing for `Qwen/Qwen3-32B` on `h200_sxm`.

| Mode | Completed | Mean TTFT | P95 TTFT | P99 TTFT | Max TTFT | Mean E2E | Output tok/s | Req/s | Prefix reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_robin` | 10,000 | 1.89 s | 4.27 s | 6.80 s | 20.07 s | 35.71 s | 816.6 | 3.46 | 0.573 |
| `sticky_session` | 10,000 | 6.02 s | 28.28 s | 39.70 s | 66.35 s | 32.77 s | 886.2 | 3.76 | 0.702 |
| `kv_router` | 10,000 | 1.64 s | 3.80 s | 6.26 s | 18.13 s | 32.30 s | 908.1 | 3.85 | 0.624 |
| `kv_router_sticky_session_proxy` | 10,000 | 1.70 s | 4.66 s | 7.50 s | 18.13 s | 30.97 s | 946.8 | 4.02 | 0.650 |

Interpretation for this trace:

- `round_robin` at CC128 is already near the desired queueing edge: mean TTFT is `1.89 s` and
  p95 TTFT is `4.27 s`.
- Hard `sticky_session` is worse than `round_robin` on TTFT despite higher prefix reuse. It pins
  sessions too rigidly and creates load imbalance.
- `kv_router` has the best TTFT in this run: mean `1.64 s`, p95 `3.80 s`.
- `kv_router_sticky_session_proxy` is not far from exact KV routing on mean TTFT, but its p95 is
  worse. It still uses the KV router load-cost path, so it is much healthier than hard sticky.
- The large max TTFT values are expected for this trace shape: output lengths are highly skewed,
  and closed-loop replay creates bursty queueing behind long decode work.

Raw reports from this run were written to:

```bash
/tmp/dynamo_replay_aiperf_10000_delta_compare_cc128
```

## What This Branch Adds

This branch adds three pieces needed to compare session-level affinity against block-level KV
affinity on a multi-turn AIPerf trace:

- `sticky_session`: assigns the first turn of each `session_id` round-robin and pins later turns to
  the same worker.
- `kv_router_sticky_session_proxy`: uses the KV router load/cost path, but replaces exact block
  overlap with a proxy assumption that a known session has a full-prefix hit on its pinned worker.
- `mooncake_delta`: an offline replay trace format that treats each Mooncake row as a per-session
  token delta, accumulates turns by `session_id`, and recomputes request block hashes from the
  cumulative token sequence.

The `mooncake_delta` path matters for this dataset because each row's `hash_ids` match that row's
own `input_length`; later turns do not include prior turns. Directly concatenating hash entries is
incorrect at turn boundaries because the last block of a turn can be partial. The implemented path
synthesizes token IDs from each row's hash IDs, appends those tokens per session, and recomputes
engine block hashes over the cumulative token vector.

## Trace Shape

The source trace is Mooncake-style JSONL with `session_id`, `input_length`, `output_length`,
`hash_ids`, and per-turn `delay`.

Full dataset shape:

- `44,000` rows
- `2,200` sessions
- `20` turns per session
- first turns have `timestamp: 0.0`
- later turns use `delay`
- output length p50/p95/p99/max is roughly `77` / `1018` / `2142` / `2999` tokens

The benchmark above uses a bounded slice:

```bash
head -n 10000 /Users/peabrane/Downloads/dataset_aiperf.jsonl > /tmp/dataset_aiperf_10000.jsonl
```

That slice contains `500` complete sessions with `20` turns per session.

Block math is exactly 64-token blocks in the source trace:

```bash
jq -r '[.input_length, (.hash_ids|length)] | @tsv' /Users/peabrane/Downloads/dataset_aiperf.jsonl \
  | awk 'function ceildiv(a,b){return int((a+b-1)/b)}
         {rows++; want=ceildiv($1,64); if($2 != want) bad++}
         END{print "rows=" rows, "exact_ceil64=" rows-bad, "bad64=" bad+0}'
```

Expected output:

```text
rows=44000 exact_ceil64=44000 bad64=0
```

Because every first turn has the same timestamp, these runs use closed-loop concurrency instead of
literal timestamp replay. Literal timestamp replay would inject all first turns at time zero.

## Environment Setup

Run from the Dynamo repository root. Use the project `.venv`; do not use system Python.

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python maturin aiconfigurator==0.8.0
.venv/bin/maturin develop --uv -m lib/bindings/python/Cargo.toml
```

If AIC database loading fails with Git-LFS pointer files or enum parse errors, install
`aiconfigurator` from a local checkout with real perf tables:

```bash
uv pip install --python .venv/bin/python --force-reinstall /path/to/aiconfigurator
.venv/bin/maturin develop --uv -m lib/bindings/python/Cargo.toml
```

The run above used:

- Python `3.13.5`
- `aiconfigurator==0.8.0`
- AIC backend `vllm`
- AIC backend version `0.19.0`
- AIC system `h200_sxm`
- AIC model path `Qwen/Qwen3-32B`
- AIC TP size `1`

## Replay Config

Common engine args:

```bash
ENGINE_ARGS='{"block_size":64,"num_gpu_blocks":16384,"enable_prefix_caching":true,"dp_size":1,"aic_backend":"vllm","aic_backend_version":"0.19.0","aic_system":"h200_sxm","aic_model_path":"Qwen/Qwen3-32B","aic_tp_size":1,"speedup_ratio":1.0}'
```

The commands below set `PYTHONPATH=lib/bindings/python/src` so the replay CLI comes from this
worktree even if another editable Dynamo package is installed in the same venv.

## Commands

```bash
TRACE=/tmp/dataset_aiperf_10000.jsonl
OUT=/tmp/dynamo_replay_aiperf_10000_delta_compare_cc128
mkdir -p "$OUT"
head -n 10000 /Users/peabrane/Downloads/dataset_aiperf.jsonl > "$TRACE"

for mode in round_robin sticky_session kv_router kv_router_sticky_session_proxy; do
  DYN_LOG='warn,dynamo_kv_router::scheduling::selector=warn' \
  PYTHONPATH=lib/bindings/python/src \
    .venv/bin/python -m dynamo.replay "$TRACE" \
      --trace-format mooncake_delta \
      --replay-mode offline \
      --router-mode "$mode" \
      --num-workers 8 \
      --replay-concurrency 128 \
      --trace-block-size 64 \
      --report-json "$OUT/${mode}.json" \
      --extra-engine-args "$ENGINE_ARGS" \
      > "$OUT/${mode}.log" 2>&1
done
```

Summarize reports:

```bash
for mode in round_robin sticky_session kv_router kv_router_sticky_session_proxy; do
  f=/tmp/dynamo_replay_aiperf_10000_delta_compare_cc128/${mode}.json
  printf '%s\t' "$mode"
  jq -r '[.completed_requests,.mean_ttft_ms,.p95_ttft_ms,.p99_ttft_ms,.max_ttft_ms,.mean_e2e_latency_ms,.output_throughput_tok_s,.request_throughput_rps,.prefix_cache_reused_ratio] | @tsv' "$f"
done
```

## Validation

Targeted replay tests:

```bash
cargo test --package dynamo-mocker accumulating_delta_mode_reblocks_partial_turns_in_token_space --lib
cargo test --package dynamo-mocker sticky_session --lib
```

Expected result:

```text
accumulating_delta_mode_reblocks_partial_turns_in_token_space ... ok
3 sticky_session tests passed
```
