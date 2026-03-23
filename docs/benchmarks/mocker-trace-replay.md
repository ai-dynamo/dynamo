---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mocker Offline Trace Replay
subtitle: Replay Mooncake-style traces offline without launching a runtime or router
---

This guide covers the mocker's offline trace replay mode, which replays a Mooncake-style JSONL trace directly through the mock scheduler and writes a metrics report. Unlike normal `dynamo.mocker` usage, this mode does not launch workers, register endpoints, or require NATS, etcd, or a frontend.

Use this when you want to:

- benchmark scheduler behavior from a saved trace
- compare timing and cache behavior across mocker configurations
- validate replay logic in CI without bringing up a distributed stack

## Quick Start

Run offline replay by passing `--trace-file`:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B
```

This writes a JSON report next to the trace file by default:

```text
/path/to/mooncake_trace.replay.json
```

The CLI also prints a `Replay Summary` table to stdout with request counts, throughput, and latency statistics.

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

The mocker synthesizes token blocks from `hash_ids` using the configured `--block-size`, so the replay block size should match the block size used when the trace was generated.

## Modes

### Fixed-Schedule Replay

Default replay mode preserves the timestamps from the trace and simulates arrivals in virtual time:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B \
    --block-size 512
```

This is the right mode when you want deterministic replay of the original arrival pattern.

### Closed-Loop Concurrency Replay

Use `--replay-concurrency` to ignore trace arrival timing and keep a fixed number of requests in flight:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B \
    --block-size 512 \
    --replay-concurrency 16
```

This mode is useful when you want to compare scheduler behavior under a fixed offered concurrency rather than the original trace schedule.

## Output

Use `--output-file` to override the default report location:

```bash
python -m dynamo.mocker \
    --trace-file /path/to/mooncake_trace.jsonl \
    --model-path Qwen/Qwen3-0.6B \
    --output-file /tmp/replay-report.json
```

If `--output-file` is not set, the report path defaults to `<trace stem>.replay.json` in the same directory as the input trace.

The report contains:

- request counts
- input and output token totals
- virtual duration and wall-clock runtime
- request and token throughput
- prefix cache reuse ratio
- TTFT, TTST, TPOT, ITL, and end-to-end latency summaries
- output-token-throughput-per-user summaries

## Replay Constraints

Offline replay currently supports only this configuration:

- `--num-workers 1`
- aggregated mode
- `--engine-type vllm`
- `--data-parallel-size 1`

If you violate those constraints, replay fails immediately with a validation error.

## Practical Notes

- `--replay-concurrency` requires `--trace-file`
- `--speedup-ratio` still affects simulated timing
- `--extra-engine-args` can be used to provide a full mocker config JSON instead of individual CLI flags
- offline replay does not need planner runtime setup, router registration, or event transport

## When To Use This vs AIPerf

Use offline replay when:

- you want a fast scheduler-only simulation
- you want deterministic CI coverage of replay behavior
- you do not need HTTP serving, frontend behavior, or network effects

Use [Dynamo Benchmarking](benchmarking.md) when:

- you want end-to-end benchmarking against a live endpoint
- you need frontend, transport, or cluster-level behavior
- you want AIPerf dashboards and endpoint-facing metrics
