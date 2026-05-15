---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AIPerf Session Routing Replay
subtitle: Reproduce round-robin, sticky-session, KV-router, and sticky-proxy replay comparisons
---

# AIPerf Session Routing Replay

## Results

| Mode | Completed | Mean TTFT | P95 TTFT | P99 TTFT | Max TTFT | Mean E2E | Output tok/s | Req/s | Prefix reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_robin` | 44,000 | 603 ms | 1.05 s | 13.52 s | 52.21 s | 15.09 s | 7,491.6 | 32.38 | 0.433 |
| `sticky_session` | 44,000 | 1,305 ms | 3.19 s | 13.52 s | 52.21 s | 15.35 s | 7,387.4 | 31.93 | 0.433 |
| `kv_router` | 44,000 | 584 ms | 815 ms | 14.58 s | 48.78 s | 14.89 s | 7,534.1 | 32.56 | 0.435 |
| `kv_router_sticky_session_proxy` | 44,000 | 592 ms | 821 ms | 14.75 s | 48.78 s | 14.91 s | 7,514.9 | 32.48 | 0.433 |

Interpretation for this trace:

- Plain sticky session is worse on mean and p95 TTFT than round-robin or KV-aware routing, even
  though the trace has strong session structure.
- Exact KV routing and the sticky-session proxy are close under this 16k-block setup.
- Exact KV has a small edge in prefix reuse and throughput, but this trace does not show a large
  gap between block-level affinity and the session proxy.
- This is consistent with a traffic shape where most deep reuse is intra-session and cross-session
  overlap is shallow.

## What This Branch Adds

This branch adds two offline replay-only router modes for comparing session-level affinity against
block-level KV affinity on a multi-turn AIPerf trace:

- `sticky_session`: assigns the first turn of each `session_id` round-robin and pins later turns to
  the same worker.
- `kv_router_sticky_session_proxy`: uses the KV router load/cost path, but replaces exact block
  overlap with a proxy assumption that a known session has a full-prefix hit on its pinned worker.

The comparison below uses the AIPerf dataset at:

```bash
/Users/peabrane/Downloads/dataset_aiperf.jsonl
```

## Trace Shape

The trace is Mooncake-style JSONL with `session_id`, `input_length`, `output_length`, `hash_ids`,
and per-turn `delay`.

Observed shape:

- `44,000` rows
- `2,200` sessions
- `20` turns per session
- first turns have `timestamp: 0.0`
- later turns use `delay`
- block math is exactly 64-token blocks:

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

The run below used:

- Python `3.13.5`
- `aiconfigurator==0.8.0`
- AIC backend `vllm`
- AIC backend version `0.19.0`
- AIC system `h200_sxm`
- AIC model path `Qwen/Qwen3-32B`
- AIC TP size `1`

## Replay Config

All four modes use the same workload and engine settings:

- replay mode: `offline`
- workers: `8`
- closed-loop concurrency: `512`
- trace block size: `64`
- engine block size: `64`
- GPU KV blocks per worker: `16,384`
- prefix caching: enabled
- engine timing: AIC-backed vLLM
- speedup ratio: `1.0`

Common engine args:

```bash
ENGINE_ARGS='{"block_size":64,"num_gpu_blocks":16384,"enable_prefix_caching":true,"dp_size":1,"aic_backend":"vllm","aic_backend_version":"0.19.0","aic_system":"h200_sxm","aic_model_path":"Qwen/Qwen3-32B","aic_tp_size":1,"speedup_ratio":1.0}'
```

## Commands

```bash
mkdir -p /tmp/dynamo_replay_aiperf_session_compare_16k

DYN_LOG=warn \
  .venv/bin/python -m dynamo.replay /Users/peabrane/Downloads/dataset_aiperf.jsonl \
  --replay-mode offline \
  --router-mode round_robin \
  --num-workers 8 \
  --replay-concurrency 512 \
  --trace-block-size 64 \
  --report-json /tmp/dynamo_replay_aiperf_session_compare_16k/round_robin.json \
  --extra-engine-args "$ENGINE_ARGS"

DYN_LOG=warn \
  .venv/bin/python -m dynamo.replay /Users/peabrane/Downloads/dataset_aiperf.jsonl \
  --replay-mode offline \
  --router-mode sticky_session \
  --num-workers 8 \
  --replay-concurrency 512 \
  --trace-block-size 64 \
  --report-json /tmp/dynamo_replay_aiperf_session_compare_16k/sticky_session.json \
  --extra-engine-args "$ENGINE_ARGS"

DYN_LOG='warn,dynamo_kv_router::scheduling::selector=warn' \
  .venv/bin/python -m dynamo.replay /Users/peabrane/Downloads/dataset_aiperf.jsonl \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 8 \
  --replay-concurrency 512 \
  --trace-block-size 64 \
  --report-json /tmp/dynamo_replay_aiperf_session_compare_16k/kv_router.json \
  --extra-engine-args "$ENGINE_ARGS"

DYN_LOG='warn,dynamo_kv_router::scheduling::selector=warn' \
  .venv/bin/python -m dynamo.replay /Users/peabrane/Downloads/dataset_aiperf.jsonl \
  --replay-mode offline \
  --router-mode kv_router_sticky_session_proxy \
  --num-workers 8 \
  --replay-concurrency 512 \
  --trace-block-size 64 \
  --report-json /tmp/dynamo_replay_aiperf_session_compare_16k/kv_router_sticky_session_proxy.json \
  --extra-engine-args "$ENGINE_ARGS"
```

Summarize reports:

```bash
for mode in round_robin sticky_session kv_router kv_router_sticky_session_proxy; do
  f=/tmp/dynamo_replay_aiperf_session_compare_16k/${mode}.json
  printf '%s\t' "$mode"
  jq -r '[.completed_requests,.mean_ttft_ms,.p95_ttft_ms,.p99_ttft_ms,.max_ttft_ms,.mean_e2e_latency_ms,.output_throughput_tok_s,.request_throughput_rps,.prefix_cache_reused_ratio] | @tsv' "$f"
done
```

## Validation

Targeted replay tests:

```bash
cargo test --package dynamo-mocker sticky_session --lib
```

Expected result:

```text
3 passed
```
