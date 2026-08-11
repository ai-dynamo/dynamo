---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AI Simulate Replay CLI Reference
subtitle: Base workload, engine, topology, SLA, and report arguments
---

> [!WARNING]
> **Experimental.** The AI Simulate replay command, configuration, and report schema can change
> without a standard deprecation period.

`python -m aisimulate.replay` runs one workload through an engine-only simulated configuration. It
uses deterministic offline replay with built-in round-robin placement and does not require
`ai-dynamo`.

Choose exactly one workload source:

- one positional trace file
- all of `--input-tokens`, `--output-tokens`, and `--request-count`

Choose exactly one engine topology:

- `--extra-engine-args` for aggregated replay
- both `--prefill-engine-args` and `--decode-engine-args` for disaggregated replay

## Example

```bash
python -m aisimulate.replay \
  --extra-engine-args '{"engine_type":"vllm","num_gpu_blocks":1024,"block_size":16,"timing_model":{"type":"fixed","prefill_ms":10,"decode_ms":2}}' \
  --input-tokens 1024 \
  --output-tokens 128 \
  --request-count 16 \
  --replay-concurrency 4
```

## Workload Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `trace_files` | positional path | none | Input trace. The standalone runner accepts one file. |
| `--input-tokens` | positive integer | none | Input length for each synthetic request. |
| `--output-tokens` | positive integer | none | Output length for each synthetic request. |
| `--request-count` | positive integer | none | Number of synthetic sessions. With one turn per session, this is also the request count. |
| `--request-rate` | positive float | none | Poisson open-loop arrival rate in requests per second. Synthetic replay only. |
| `--arrival-interval-ms` | nonnegative float | none | Fixed open-loop interval between synthetic arrivals. |
| `--arrival-seed` | integer | `42` | Random seed for Poisson synthetic arrivals. |
| `--turns-per-session` | positive integer | `1` | Turns generated for each synthetic session. |
| `--shared-prefix-ratio` | float from `0` to `1` | `0` | Fraction of prompt blocks shared within each synthetic prefix group. |
| `--num-prefix-groups` | nonnegative integer | `0` | Synthetic prefix groups. `0` disables grouping. |
| `--inter-turn-delay-ms` | nonnegative float | `0` | Delay after one synthetic turn completes before the next becomes eligible. |
| `--replay-concurrency` | positive integer | none | Closed-loop in-flight cap. Required for synthetic replay unless an open-loop arrival control is selected. |
| `--arrival-speedup-ratio` | positive float | `1` | Compresses or stretches trace arrival time without changing engine timing. |
| `--max-sim-time-seconds` | nonnegative float | none | Stops trace admission at the specified virtual time. Trace replay only. |

Synthetic replay requires exactly one of `--replay-concurrency`, `--request-rate`, or
`--arrival-interval-ms`.

## Trace Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--trace-format` | enum | `mooncake` | Selects `mooncake`, `mooncake-delta`, `agentic_mooncake`, `applied_compute_agentic`, or `dynamo`. |
| `--trace-block-size` | positive integer | unset | Tokens represented by each source `hash_id`. Dynamo traces can carry this value in the trace. |
| `--trace-shared-prefix-ratio` | float from `0` to `1` | `0` | Overrides or synthesizes shared-prefix structure for supported trace inputs. |
| `--trace-num-prefix-groups` | nonnegative integer | `0` | Number of trace prefix groups when prefix synthesis is enabled. |

Mooncake-compatible JSONL rows accept request arrival time, input and output length, optional prefix
hashes, and optional session or dependency fields. A basic row is:

```json
{"timestamp":0,"input_length":2048,"output_length":128,"hash_ids":[0,1,2,3]}
```

Rows with the same `session_id` run as ordered turns. `agentic_mooncake` rows can add stable
`request_id`, `wait_for`, `branches`, `prefix_reset`, `delay`, and `tool_wait_ms` fields.

The `dynamo` format reads `dynamo.request.trace.v1` JSONL or JSONL.GZ. The shared parser accepts
multiple shards for the Dynamo-integrated command, but the standalone AI Simulate runner currently
requires exactly one trace file.

## Engine and Topology Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--extra-engine-args` | JSON object | none | Aggregated engine configuration. Mutually exclusive with prefill and decode engine arguments. |
| `--prefill-engine-args` | JSON object | none | Prefill engine configuration for disaggregated replay. Requires `--decode-engine-args`. |
| `--decode-engine-args` | JSON object | none | Decode engine configuration for disaggregated replay. Requires `--prefill-engine-args`. |
| `--num-workers` | positive integer | `1` | Logical aggregated workers. |
| `--num-prefill-workers` | positive integer | `1` | Logical prefill workers in a disaggregated topology. |
| `--num-decode-workers` | positive integer | `1` | Logical decode workers in a disaggregated topology. |
| `--replay-mode` | enum | `offline` | Shared parser mode. `python -m aisimulate.replay` rejects `online`; online replay requires Dynamo. |

Engine JSON selects `engine_type` as `vllm`, `sglang`, or `trtllm`. Common fields include
`block_size`, `num_gpu_blocks`, `max_num_seqs`, `max_num_batched_tokens`, `dp_size`, prefix-caching
controls, and `timing_model`. The selected backend validates backend-specific fields.

For NVIDIA AI Configurator timing, engine JSON can identify the backend, version, system, model,
parallelism, quantization, and speculative-decoding settings. These values configure engine timing;
they do not enable Dynamo Router-side load estimation.

Prefill and decode configurations must select the same backend and compatible backend versions.
Disaggregated attention-DP is not supported.

## SLA Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--sla-ttft-ms` | nonnegative float | none | Maximum Time To First Token (TTFT) for goodput. |
| `--sla-itl-ms` | nonnegative float | none | Maximum average Inter Token Latency (ITL) for goodput. |
| `--sla-e2e-ms` | nonnegative float | none | Maximum end-to-end latency for goodput. |

Set TTFT and ITL together to gate both prompt and decode latency, or set end-to-end latency as a
single request-level target.

## Output Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--report-json` | path | timestamped file | Writes the full report. The default name starts with `aisimulate_replay_report_`. |
| `--per-request-jsonl` | path | none | Captures and writes one terminal record per request. |

The report contains request and token counts, virtual duration, wall-clock execution time,
throughput, prefix-cache reuse, latency distributions, optional SLA goodput, and optional execution
evidence. Latency distributions include TTFT, Time To Second Token (TTST), Time Per Output Token
(TPOT), ITL, and end-to-end latency.

## Dynamo Extension

`python -m dynamo.replay` uses this same base parser and adds KV router, Planner, Router-side NVIDIA AI
Configurator, model-profile, and online-runtime arguments. Those options and their constraints remain
part of the Dynamo replay interface.
