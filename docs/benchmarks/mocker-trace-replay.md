---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mocker Trace Replay
subtitle: Replay Mooncake-style traces through the mocker in offline or online mode
---

This guide covers trace replay support for Mooncake-style JSONL traces via `python -m dynamo.replay`,
which prints an AIPerf-style summary table, writes the full replay report JSON to disk, and exposes
`offline|online`, `round_robin|kv_router`, `arrival_speedup_ratio`, closed-loop concurrency, and
synthetic workload inputs directly.

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
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}' \
    --report-json /tmp/replay-report.json
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
    --extra-engine-args '{"block_size":512}' \
    --report-json /tmp/replay-report.json
```

Run synthetic workload replay when you want shared-prefix or multi-turn structure without a trace
file:

```bash
python -m dynamo.replay \
    --input-tokens 5000 \
    --output-tokens 500 \
    --request-count 200 \
    --turns-per-session 3 \
    --shared-prefix-ratio 0.5 \
    --num-prefix-groups 8 \
    --inter-turn-delay-ms 250 \
    --replay-mode offline \
    --replay-concurrency 32 \
    --extra-engine-args '{"block_size":512}' \
    --report-json /tmp/replay-report.json
```

`python -m dynamo.replay` prints an AIPerf-style summary table to stdout and writes the full replay
report JSON to disk.

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

Replay also supports multi-turn sessions. Use the same `session_id` on all turns in a session. The
first turn uses `timestamp` or `created_time`; later turns may use either:

- `delay` or `delay_ms` directly
- or an absolute later `timestamp`, in which case replay infers the inter-turn delay from the
  previous turn timestamp

Example:

```json
{"session_id":"session-a","timestamp":1000,"input_length":2048,"output_length":128,"hash_ids":[1,2,3,4]}
{"session_id":"session-a","delay":250,"input_length":2560,"output_length":128,"hash_ids":[1,2,3,4,5]}
{"session_id":"session-b","timestamp":1010,"input_length":1024,"output_length":64,"hash_ids":[9,10]}
{"session_id":"session-b","delay_ms":50,"input_length":1536,"output_length":64,"hash_ids":[9,10,11]}
```

Replay uses two different block-size concepts for trace files:

- `--trace-block-size`: how many tokens each `hash_id` in the dataset represents
- engine `block_size`: the block size used by the replay engine and router when they re-chunk the
  synthesized tokens into sequence hashes

Public Mooncake/toolagent traces use `512` tokens per `hash_id`, so replaying them should normally
use `--trace-block-size 512`. The engine `block_size` can still be smaller, for example the live
vLLM benchmark setup uses `block_size=64`. For `engine_type=sglang`, replay still uses canonical
`block_size` internally; `sglang.page_size` is accepted as a compatibility alias and is normalized
into `block_size` before replay starts.

## Replay Surfaces

### `python -m dynamo.replay`

The dedicated replay CLI exposes:

- either a positional `trace_file`, or all of `--input-tokens`, `--output-tokens`, and `--request-count`
- `--replay-mode offline|online`
- `--router-mode round_robin|kv_router`
- `--num-workers`
- `--num-prefill-workers`
- `--num-decode-workers`
- `--replay-concurrency`
- `--arrival-interval-ms`
- `--arrival-speedup-ratio`
- `--trace-block-size`
- `--turns-per-session`
- `--shared-prefix-ratio`
- `--num-prefix-groups`
- `--inter-turn-delay-ms`
- `--extra-engine-args` (JSON string)
- `--prefill-engine-args` (JSON string)
- `--decode-engine-args` (JSON string)
- `--router-config` (JSON string)
- `--aic-backend`
- `--aic-system`
- `--aic-backend-version`
- `--aic-tp-size`
- `--aic-model-path`
- `--report-json`

Defaults:

- `--replay-mode offline`
- `--router-mode round_robin`

Example:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode online \
    --router-mode kv_router \
    --num-workers 4 \
    --arrival-speedup-ratio 10 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}' \
    --router-config '{"router_queue_policy":"fcfs","router_temperature":0.0}' \
    --report-json /tmp/replay-report.json
```

SGLang replay uses the same CLI surface. A minimal extra-engine-args file can use either
`block_size` directly or the compatibility alias `sglang.page_size`:

```json
{
  "engine_type": "sglang",
  "num_gpu_blocks": 512,
  "sglang": {
    "page_size": 2
  }
}
```

Both `--extra-engine-args` and `--router-config` accept partial JSON objects. Engine settings such
as `block_size`, `engine_type`, `dp_size`, `speedup_ratio`, and `decode_speedup_ratio` belong in
`--extra-engine-args`, not as top-level replay CLI flags. `--trace-block-size` is separate and is
used only for trace-file replay. Unspecified fields fall back to the same defaults used by
`MockEngineArgs::default()` and `KvRouterConfig::default()`.

Replay has two independent AIC surfaces:

- engine timing AIC via `--extra-engine-args` / staged engine JSON
- router-side prompt-load AIC via top-level `--aic-*` flags together with
  `router_prefill_load_model: "aic"` in `--router-config`

Offline disagg replay uses staged engine args instead of `--extra-engine-args`:

- `--prefill-engine-args` for the prefill worker config
- `--decode-engine-args` for the decode worker config
- `--num-prefill-workers` and `--num-decode-workers` for pool sizes

For offline disagg replay, the staged JSON must set `worker_type` explicitly:

- `--prefill-engine-args` must use `worker_type: "prefill"`
- `--decode-engine-args` must use `worker_type: "decode"`

The staged configs must also use the same engine `block_size`. `--trace-block-size` remains a
separate trace-file input knob.

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
    --extra-engine-args '{"block_size":512}'
```

This is useful for parameter sweeps where Mooncake-style prefix structure is not required.

When `--turns-per-session > 1`, `--request-count` is interpreted as the number of sessions rather
than the total number of emitted turns. The total completed request count becomes:

- `request_count * turns_per_session`

Synthetic workload options:

- `--turns-per-session`: number of turns in each synthetic session
- `--shared-prefix-ratio`: fraction of prompt blocks shared inside a prefix group
- `--num-prefix-groups`: number of shared-prefix groups; `0` disables grouping
- `--inter-turn-delay-ms`: constant delay applied after each completed turn before the next turn in
  the same session becomes eligible

## Modes

### Fixed-Schedule Replay

Default trace replay preserves the timestamps from the trace and simulates arrivals according to
those timestamps:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}'
```

This is the right mode when you want deterministic replay of the original arrival pattern.

### Closed-Loop Concurrency Replay

Use `--replay-concurrency` to ignore first-turn trace arrival timing and keep a fixed number of
requests in flight:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --replay-concurrency 16
```

This mode is useful when you want to compare scheduler behavior under a fixed offered concurrency rather than the original trace schedule.

For multi-turn sessions, concurrency mode still enforces session order and inter-turn delays:

- first-turn timestamps are ignored
- turn `n+1` is not eligible until turn `n` completes
- `delay` / `delay_ms` / synthetic `--inter-turn-delay-ms` are still applied after completion
- TTFT is measured from actual dispatch under the cap, not from the ignored trace timestamp

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
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}'
```

### Arrival Speedup

Use `--arrival-speedup-ratio` to compress or stretch the trace arrival process without changing the
mocker compute model. Larger values make arrivals happen sooner relative to the original trace.

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --num-workers 4 \
    --arrival-speedup-ratio 5 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}'
```

### Router Modes

Replay currently supports:

- `round_robin`
- `kv_router`

`kv_router` uses the shared local scheduler and an in-process KV indexer. Router policy tuning is
provided through `--router-config`, not a dedicated top-level replay flag. In offline replay:

- `kv_router` is supported only when `num_workers > 1`
- router queueing is enabled and uses simulation time rather than wall-clock time
- KV visibility is delayed slightly relative to request lifecycle events
- queue admission is driven by router lifecycle edges (`add_request`, `mark_prefill_completed`, and `free`)
- transient in-pass prefill occupancy is still approximated at the router level rather than modeled exactly
- when `router_prefill_load_model` is `"aic"`, replay predicts one expected prefill duration per
  admitted request and decays only the oldest active prefill request on each worker

To compare queue policies manually, keep the same trace and engine args fixed and swap only
`router_queue_policy` inside `--router-config`:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --router-mode kv_router \
    --num-workers 4 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}' \
    --router-config '{"router_queue_policy":"fcfs"}'

python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --router-mode kv_router \
    --num-workers 4 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}' \
    --router-config '{"router_queue_policy":"lcfs"}'
```

`lcfs` is intentionally a worse comparison policy under saturation; use it for experiments, not as
an expected production default.

To enable router-side AIC prefill-load modeling during replay:

```bash
python -m dynamo.replay /path/to/mooncake_trace.jsonl \
    --replay-mode offline \
    --router-mode kv_router \
    --num-workers 4 \
    --trace-block-size 512 \
    --extra-engine-args '{"block_size":64}' \
    --router-config '{"router_track_prefill_tokens":true,"router_prefill_load_model":"aic"}' \
    --aic-backend vllm \
    --aic-system h200_sxm \
    --aic-model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --aic-tp-size 1
```

For offline disagg replay, the same top-level `--aic-*` flags are supported, but the estimator is
applied only to the prefill-stage router.

## Output

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

If `--report-json` is not provided, `python -m dynamo.replay` writes a timestamped
`dynamo_replay_report_*.json` file in the current working directory.

## Replay Constraints

Shared replay constraints:

- `extra_engine_args.engine_type` must be `vllm` or `sglang`
- aggregated replay requires the existing aggregated args path
- disagg replay requires both `prefill_engine_args` and `decode_engine_args`
- disagg replay requires `router_mode=kv_router`
- replay `dp_size` must be `1`
- disagg replay requires matching `block_size` in `prefill_engine_args` and `decode_engine_args`

Additional offline constraints:

- offline `kv_router` requires `num_workers > 1`
- single-worker offline replay is still a dedicated fast path for `vllm`, but it now supports both
  flat request replay and workload-driven multi-turn replay
- `sglang` still goes through the shared multi-worker replay runtime even when `num_workers=1`
- offline disagg replay is a separate two-stage runtime with prefill and decode worker pools

Additional online constraints:

- the current live replay path is also limited to aggregated workers

If you violate those constraints, replay fails immediately with a validation error.

## Practical Notes

- `python -m dynamo.replay` requires exactly one of:
  either a trace file, or all of `--input-tokens`, `--output-tokens`, and `--request-count`
- `--replay-concurrency` works with both trace replay and synthetic replay
- mocker compute-speed knobs such as `speedup_ratio` still affect simulated timing when passed via
  the engine-args JSON for the chosen replay mode
- `--arrival-speedup-ratio` affects trace timestamps, not worker compute speed
- `--trace-block-size` affects only how trace `hash_ids` expand into tokens
- `--arrival-interval-ms` only applies to synthetic replay
- `--turns-per-session`, `--shared-prefix-ratio`, `--num-prefix-groups`, and
  `--inter-turn-delay-ms` only apply to synthetic replay
- `--extra-engine-args`, `--prefill-engine-args`, `--decode-engine-args`, and `--router-config`
  are JSON strings on the standalone replay CLI
- top-level `--aic-*` flags are used only for router-side prompt-load modeling; engine timing AIC
  still belongs in the engine-args JSON
- offline replay does not need planner runtime setup, router registration, or external event transport
- trace-file replay can use different values for `--trace-block-size` and engine `block_size`
- Mooncake/toolagent traces typically use `--trace-block-size 512`, while engine `block_size`
  often stays `64`

## When To Use This vs AIPerf

Use offline replay when:

- you want a fast scheduler-only simulation
- you want deterministic CI coverage of replay behavior
- you do not need HTTP serving, frontend behavior, or network effects

Use [Dynamo Benchmarking](benchmarking.md) when:

- you want end-to-end benchmarking against a live endpoint
- you need frontend, transport, or cluster-level behavior
- you want AIPerf dashboards and endpoint-facing metrics
