---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Simulation
subtitle: Convert agent request traces into replay and simulation inputs
---

Use agent simulation to replay collected agent trajectories against mock Dynamo workers. Start with [Agent Tracing](agent-tracing.md) to collect request rows, then convert those rows into an agentic Mooncake trace for `python -m dynamo.replay`.

## Collect a Trace

Enable request tracing while running the agent workload:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_SINKS=jsonl_gz
export DYN_REQUEST_TRACE_OUTPUT_PATH=/tmp/dynamo-trace
```

For tool timing fidelity, publish explicit tool events over the optional ZMQ ingress described in [Agent Tracing](agent-tracing.md#tool-call-observability). Without tool events, Dynamo can still infer tool-wait time from the gap between adjacent LLM requests in the same session.

## Convert to Agentic Mooncake

**Experimental.** The converter uses Dynamo `request_end` rows for request timing, token lengths, worker placement, and replay hashes. It also uses terminal harness tool rows (`tool_end` / `tool_error`) to preserve tool-wait time between dependent LLM requests.

Replay ignores non-replay request fields such as `finish_reason_metadata`; use the Perfetto view in [Agent Tracing](agent-tracing.md#view-traces-in-perfetto) when you want to inspect final finish reasons, backend stop signals, or complete tool-call metadata inside the trace.

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --agentic \
  --input-path "${DYN_REQUEST_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output-file /tmp/dynamo-request-trace.agentic-mooncake.jsonl
```

## Replay Offline

The binary prints `trace_block_size`. Use that exact value for replay so hash segmentation matches what Dynamo recorded. Align the mock engine block size with the same number in `--extra-engine-args`.

```bash
TRACE_BLOCK_SIZE=128
uv run --no-sync python -m dynamo.replay /tmp/dynamo-request-trace.agentic-mooncake.jsonl \
  --trace-format agentic_mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

`kv_router` needs at least two mock workers. For a single-worker smoke test, use `--router-mode round_robin --num-workers 1`.

## Agentic Row Semantics

Agentic Mooncake rows preserve:

- `request_id`: the LLM request row identity.
- Mooncake `session_id`: derived from the Dynamo `session_id`.
- `wait_for`: request IDs that must complete before this row becomes eligible.
- `branches`: child request IDs spawned from this row.
- `prefix_reset`: first request in a session.
- `delay`: non-tool delay after dependencies finish.
- `tool_wait_ms`: tool time after dependencies finish, parallel-aware as the union of overlapping spans rather than their sum.
- `tool_events`: per-tool spans attributed to this LLM request, each carrying `tool_call_id`, `tool_class`, `status`, `started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`, and optional `output_bytes`, `output_tokens`, or `error_type`.
- `hash_ids`, `input_length`, and `output_length`: prompt-prefix and length data for mocker replay.

Rows with no `wait_for` use their `timestamp` as the replay start time. Rows with dependencies wait for all listed requests to complete, then wait `delay + tool_wait_ms` before dispatch. For more flags and engine settings, see [DynoSim Runs](../dynosim/runs.md).