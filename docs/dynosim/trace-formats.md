---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Trace Format Reference
sidebar-title: Trace Formats
subtitle: Inputs, converters, and replay constraints for DynoSim and AIPerf
---

## DynoSim Trace Formats

Set `--trace-format` to one of these values:

| Format | Workload Model | Supported Scheduling | Constraints |
|---|---|---|---|
| `mooncake` | Timestamped requests or closed-loop sessions with cumulative prompt hashes | Fixed schedule or `--replay-concurrency`; offline aggregated or disaggregated; online aggregated | Each row describes the full prompt shape |
| `mooncake-delta` | Per-session prompt deltas accumulated during replay | Offline aggregated; fixed schedule or `--replay-concurrency` | Not supported online or with disaggregated replay; DynoSim accumulates each session prompt in memory |
| `agentic_mooncake` | Request DAG with sequence, spawn, join, and tool-delay edges | Offline aggregated fixed schedule | Not supported online, with disaggregated replay, or with `--replay-concurrency` |
| `applied_compute_agentic` | Sequential sessions synthesized from turn-length and tool-latency arrays | Concurrency-driven replay | Requires `--replay-concurrency`; source rows have no first-turn timestamps |

## Mooncake

Each JSONL row describes one full prompt:

```json
{"session_id":"session-a","timestamp":1000,"input_length":2048,"output_length":128,"hash_ids":[1,2,3,4]}
```

Required replay fields are `output_length` and `hash_ids`. Set `input_length` when the final hash
block is only partly filled. Use `timestamp` or `created_time` for arrival time. Rows with the same
`session_id` form a closed-loop session; later rows can use `delay` or `delay_ms` instead of a
timestamp.

`--trace-block-size` tells DynoSim how many tokens each `hash_id` represents. It is separate from
the mock engine `block_size`, which controls scheduler and KV router block boundaries.

## Mooncake Delta

Mooncake Delta uses the Mooncake row shape, but each turn contains only the newly added prompt
segment. DynoSim appends those tokens to the session's prior prompt before dispatch. Use it when the
source stores compact turn deltas rather than cumulative prompts.

Do not use this format for independent rows that already contain full prompts. The replay would
append unrelated prompt content.

## Agentic Mooncake

Agentic Mooncake adds request identity and dependency fields to Mooncake rows:

| Field | Meaning |
|---|---|
| `request_id` | Stable identity for one LLM request |
| `session_id` | Session identity |
| `wait_for` | Request IDs that must complete before this request starts |
| `branches` | Child-request metadata; does not control replay scheduling |
| `prefix_reset` | Session-boundary metadata; does not control replay scheduling |
| `delay` | Non-tool delay after dependencies complete |
| `tool_wait_ms` | Tool execution time after dependencies complete |
| `tool_events` | Per-tool analysis metadata; replay scheduling uses `tool_wait_ms` |

Rows without dependencies use `timestamp` as their start time. Rows with `wait_for` start after all
listed requests complete, followed by `delay + tool_wait_ms`.

Convert a Dynamo request trace with:

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --agentic \
  --input-path /tmp/dynamo-agent-trace.000000.jsonl.gz \
  --output-file /tmp/dynamo-agent-trace.agentic-mooncake.jsonl
```

Every converted request row must have `agent_context`. The converter rejects mixed context-free and
agent-aware request rows.

## Applied Compute Agentic

DynoSim accepts the
[Applied Compute TRIE workload format](https://github.com/Applied-Compute/trie/blob/main/README.md#workload-format).
Each JSONL row describes one sequential agent session:

```json
{
  "num_turns": 2,
  "input_prompt_length": 4096,
  "assistant_response_length": [128, 96],
  "tool_call_output_length": [512, 256],
  "tool_call_latency": [1.4, 0.8],
  "final_assistant_response_length": 64
}
```

The three arrays must each contain `num_turns` values. `tool_call_latency` is measured in seconds.
DynoSim adds each assistant response and tool output to the next turn's prompt, then appends one
final assistant request.

Run this format with `--replay-concurrency`. Use `--trace-shared-prefix-ratio` and
`--trace-num-prefix-groups` only when you want to synthesize cross-session prefix reuse.

## Coding-Agent and Text Inputs

The local Claude exporter writes ordinary Mooncake plus a text-free sidecar:

```bash
cargo run -p dynamo-bench --bin claude_trace_export -- \
  --output-file /tmp/claude-trace.mooncake.jsonl
```

DynoSim consumes the Mooncake file, not the sidecar. See the
[coding trace exporter reference](https://github.com/ai-dynamo/dynamo/blob/main/lib/bench/coding/README.md)
for discovery and tokenizer options.

Local Codex and OpenCode exporters do not exist. Their live requests still gain session context
when sent through Dynamo with the headers listed in
[Session IDs](../agents/session-ids.md#session-id-inputs).

No generic text-to-Mooncake command exists. New adapters can reuse `MooncakeRow`,
`MooncakeJsonlWriter`, and the rolling-hash helpers in
[`lib/data-gen`](https://github.com/ai-dynamo/dynamo/tree/main/lib/data-gen). AIPerf provides the
same building block through
[`RollingHasher`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/dataset/synthesis/rolling_hasher.py).
