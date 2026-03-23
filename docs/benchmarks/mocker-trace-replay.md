---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mocker Trace Replay
subtitle: Replay Mooncake-style traces through the mocker in offline or online mode
---

This guide covers the mocker's trace replay support for Mooncake-style JSONL traces. The replay
surface is available in two forms:

- `python -m dynamo.mocker --trace-file ...`, which writes a report file and prints a replay summary
- `python -m dynamo.replay ...`, which returns the replay report JSON on stdout and exposes
  `offline|online`, `round_robin|kv_router`, `arrival_speedup_ratio`, and synthetic replay inputs
  directly

Unlike normal `dynamo.mocker` usage, offline replay does not launch workers, register endpoints, or
require NATS, etcd, or a frontend. Online replay does exercise the live mock-worker runtime path.

Use this when you want to:

- benchmark scheduler behavior from a saved trace
- compare timing and cache behavior across mocker configurations
- validate replay logic in CI without bringing up a distributed stack

## Quick Start

Run offline replay through the dedicated replay CLI:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --num-workers 4 \
    --replay-mode offline \
    --router-mode round_robin \
    --extra-engine-args /path/to/mocker_args.json
```

Run synthetic replay through the same CLI when you want fixed request shapes without a trace file:

```bash
python -m dynamo.replay \
    --input-tokens 5000 \
    --output-tokens 500 \
    --request-count 1000 \
    --arrival-interval-ms 1.0 \
    --num-workers 1 \
    --replay-mode offline \
    --replay-concurrency 100 \
    --extra-engine-args /path/to/mocker_args.json
```

You can also run replay through the mocker CLI by passing `--trace-file`:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B
```

This writes a JSON report next to the trace file by default:

```text
/path/to/mooncake_trace.replay.json
```

`python -m dynamo.replay` prints the replay report JSON directly to stdout. The mocker CLI prints a
`Replay Summary` table to stdout and writes the report JSON to disk.

## Input Format

The trace file must be Mooncake-style JSONL. Each line should contain:

- `timestamp` or `created_time`
- `input_length` or `input_tokens`
- `output_length` or `output_tokens`
- `hash_ids`

Example:

```json
{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3]}
```

The mocker synthesizes token blocks from `hash_ids` using the configured `--block-size`, so the
replay block size must match the block size used when the trace was generated. Public Mooncake
traces are commonly block-level hashes at `512` tokens per hash ID, so replaying them with the
default mocker `block_size=64` will fail once `input_length > len(hash_ids) * 64`. For
`engine_type=sglang`, replay still uses canonical `block_size` internally; `sglang.page_size` is
accepted as a compatibility alias and is normalized into `block_size` before replay starts.

## Replay Surfaces

### `python -m dynamo.replay`

The dedicated replay CLI exposes:

- either a positional `trace_file`, or all of `--input-tokens`, `--output-tokens`, and `--request-count`
- `--replay-mode offline|online`
- `--router-mode round_robin|kv_router`
- `--router-queue-policy fcfs|wspt|lcfs`
- `--num-workers`
- `--replay-concurrency`
- `--arrival-interval-ms`
- `--arrival-speedup-ratio`
- `--extra-engine-args`
- `--extra-engine-args-json`
- `--router-config`
- `--router-config-json`

Example:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode online \
    --router-mode kv_router \
    --num-workers 4 \
    --arrival-speedup-ratio 10 \
    --extra-engine-args-json '{"block_size":64,"speedup_ratio":1000.0}' \
    --router-config-json '{"router_queue_policy":"fcfs","router_temperature":0.0}'
```

SGLang replay uses the same CLI surface. A minimal extra-engine-args file can use either
`block_size` directly or the compatibility alias `sglang.page_size`:

```json
{
  "engine_type": "sglang",
  "num_gpu_blocks": 512,
  "speedup_ratio": 1000.0,
  "sglang": {
    "page_size": 2
  }
}
```

For both `--extra-engine-args-json` and `--router-config-json`, replay accepts partial JSON
objects. Unspecified fields fall back to the same defaults used by `MockEngineArgs::default()`
and `KvRouterConfig::default()`.

### `python -m dynamo.mocker --trace-file`

The mocker CLI supports offline replay and remains useful when you want the historical
`Replay Summary` output and report-file workflow.

### Synthetic Replay

Synthetic replay bypasses trace loading and generates in-memory requests with fixed input/output
lengths and optional synthetic arrival spacing:

```bash
python -m dynamo.replay \
    --input-tokens 5000 \
    --output-tokens 500 \
    --request-count 200 \
    --arrival-interval-ms 0.5 \
    --replay-mode offline \
    --replay-concurrency 50 \
    --extra-engine-args /path/to/mocker_args.json
```

This is useful for parameter sweeps where Mooncake-style prefix structure is not required.

## Modes

### Fixed-Schedule Replay

Default trace replay preserves the timestamps from the trace and simulates arrivals according to
those timestamps:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --extra-engine-args /path/to/mocker_args.json
```

This is the right mode when you want deterministic replay of the original arrival pattern.

### Closed-Loop Concurrency Replay

Use `--replay-concurrency` to ignore trace arrival timing and keep a fixed number of requests in
flight:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --replay-concurrency 16
```

This mode is useful when you want to compare scheduler behavior under a fixed offered concurrency rather than the original trace schedule.

### Online Replay

Online replay launches the mock workers and replays the trace against the live runtime path. This
is useful when you want the replay to include live request dispatch, live output handling, and the
same async KV-event propagation model used by the current router integration.

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode online \
    --router-mode kv_router \
    --num-workers 4 \
    --arrival-speedup-ratio 10 \
    --extra-engine-args /path/to/mocker_args.json
```

### Arrival Speedup

Use `--arrival-speedup-ratio` to compress or stretch the trace arrival process without changing the
mocker compute model. Larger values make arrivals happen sooner relative to the original trace.

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --arrival-speedup-ratio 5 \
    --extra-engine-args /path/to/mocker_args.json
```

### Router Modes

Replay currently supports:

- `round_robin`
- `kv_router`

`kv_router` uses the shared local scheduler and an in-process KV indexer. In offline replay:

- `kv_router` is supported only when `num_workers > 1`
- router queueing is enabled and uses simulation time rather than wall-clock time
- KV visibility is delayed slightly relative to request lifecycle events
- queue admission is driven by router lifecycle edges (`add_request`, `mark_prefill_completed`, and `free`)
- transient in-pass prefill occupancy is still approximated at the router level rather than modeled exactly

To compare queue policies manually, keep the same trace and engine args fixed and swap only
`--router-queue-policy`:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --router-mode kv_router \
    --router-queue-policy fcfs \
    --num-workers 4 \
    --extra-engine-args /path/to/mocker_args.json

python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --router-mode kv_router \
    --router-queue-policy lcfs \
    --num-workers 4 \
    --extra-engine-args /path/to/mocker_args.json
```

`lcfs` is intentionally a worse comparison policy under saturation; use it for experiments, not as
an expected production default.

## Output

Use `--output-file` to override the default report location:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B \
    --output-file /tmp/replay-report.json
```

If `--output-file` is not set, the report path defaults to `TRACE_STEM.replay.json` in the same directory as the input trace.

The report contains:

- request counts
- input and output token totals
- virtual duration and wall-clock runtime
- request and token throughput
- prefix cache reuse ratio
- TTFT, TTST, TPOT, ITL, and end-to-end latency summaries
- output-token-throughput-per-user summaries

The dedicated replay CLI returns the same report schema as the Python APIs
`dynamo.replay.run_trace_replay(...)` and `dynamo.replay.run_synthetic_trace_replay(...)`.

## Replay Constraints

Shared replay constraints:

- aggregated mode
- `--engine-type vllm|sglang`
- `--data-parallel-size 1`

Additional offline constraints:

- offline `kv_router` requires `num_workers > 1`
- public single-worker offline replay still uses the legacy single-worker runtime for `vllm`
  while `sglang` goes through the shared multi-worker replay runtime even when `num_workers=1`

Additional online constraints:

- the current live replay path is also limited to aggregated workers

If you violate those constraints, replay fails immediately with a validation error.

## Practical Notes

- `python -m dynamo.replay` requires exactly one of:
  either a trace file, or all of `--input-tokens`, `--output-tokens`, and `--request-count`
- `--replay-concurrency` works with both trace replay and synthetic replay
- `--speedup-ratio` still affects simulated timing
- `--arrival-speedup-ratio` affects trace timestamps, not worker compute speed
- `--arrival-interval-ms` only applies to synthetic replay
- `--extra-engine-args` can be used to provide a full mocker config JSON instead of individual CLI flags
- offline replay does not need planner runtime setup, router registration, or external event transport
- the replay block size should match the trace block size, because token synthesis expands `hash_ids`
  using the configured block size

## When To Use This vs AIPerf

Use offline replay when:

- you want a fast scheduler-only simulation
- you want deterministic CI coverage of replay behavior
- you do not need HTTP serving, frontend behavior, or network effects

Use [Dynamo Benchmarking](benchmarking.md) when:

- you want end-to-end benchmarking against a live endpoint
- you need frontend, transport, or cluster-level behavior
- you want AIPerf dashboards and endpoint-facing metrics
