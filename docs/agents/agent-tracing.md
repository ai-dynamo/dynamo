---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Export Dynamo request traces, tool-call metadata, Perfetto timelines, and replay inputs
---

Agent tracing records what Dynamo measured for each eligible LLM request. When a request carries [trajectory identity](trajectory-ids.md), trace rows include the trajectory fields so you can join LLM requests, inferred tool calls, optional harness tool spans, Perfetto slices, and replay artifacts.

Tracing is best-effort profiling data, not an audit log. Dynamo does not store tool-call arguments in request traces. Use audit sinks when you need request or response payloads.

## Enable Output

The fast path is one environment variable:

```bash
export DYN_REQUEST_TRACE=1
```

That selects `jsonl_gz` output at `/tmp/dynamo-request-trace.*.jsonl.gz`. Tool-call understanding works immediately from `request_end` finish metadata: no harness tooling and no sockets are required. The optional ZMQ tool-event ingress is opt-in; see [Tool Call Observability](#tool-call-observability).

To relocate captures, set an output path:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_OUTPUT_PATH=/mnt/captures/run-42/request-trace
```

`DYN_REQUEST_TRACE` is the only trace switch. The same request trace stream contains compact replay rows when no trajectory identity is present and enriched agent rows when it is. All request trace variables are documented in [Request Replay Tracing](../observability/request-tracing.md).

## Dynamo `request_end` Record

Dynamo emits `request_end` after the response stream finishes or is dropped. The record carries trajectory identity, `output_tokens`, and autodetected `finish_reason_metadata` such as tool-call names and finish reasons. `request_id` correlates with audit rows. The `replay` block feeds Mooncake replay when Dynamo can represent the request as one replay row. Tool-call metadata is IDs and names only; arguments are intentionally not stored.

<details>
<summary>Full <code>request_end</code> record</summary>

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "trajectory_id": "research-run-42:researcher",
    "parent_trajectory_id": "research-run-42:planner"
  },
  "request": {
    "request_id": "dynamo-request-id",
    "model": "my-model",
    "output_tokens": 16,
    "finish_reason_metadata": {
      "finish_reason": "tool_calls",
      "backend_finish_reason": "stop",
      "stop_reason": "END",
      "tool_calls": [
        {
          "choice_index": 0,
          "tool_call_index": 0,
          "id": "call-abc",
          "name": "web_search"
        }
      ],
      "choices": [
        {
          "choice_index": 0,
          "finish_reason": "tool_calls",
          "backend_finish_reason": "stop",
          "stop_reason": "END"
        }
      ]
    },
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [14879255164371896291, 274632075616497421]
    }
  }
}
```

`finish_reason_metadata` is optional. `finish_reason` is the final OpenAI-compatible reason after parser rewrites, such as `tool_calls`. `backend_finish_reason` and `stop_reason` come from the backend stop path. Top-level finish fields summarize the emitted single-choice request row.

Current request tracing skips unsupported multi-choice replay shapes such as `n > 1` and `best_of > 1`, so do not assume every trajectory turn is present unless skipped-row warnings are absent. For chat streams, finish metadata is recorded after parser and jail rewrites. Completion streams record the final OpenAI-compatible completion finish reason. See the [request trace schema source](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/request_trace/types.rs) for the preferred Rust schema.

</details>

## Tool Call Observability

Default behavior requires no harness work. Dynamo parses each response stream and records the tool calls the model made into [`request_end.finish_reason_metadata`](#dynamo-request_end-record): the per-turn `finish_reason` and each call's `name` and `id`. Arguments are never stored. This is active whenever `DYN_REQUEST_TRACE=1` and the worker runs a tool-call parser with `--dyn-tool-call-parser`.

You can recover tool-wait time offline without tool events. Within a trajectory, the agent is sequential, so the gap between one turn finishing and the next arriving is the tool plus agent-overhead time:

```text
tool_wait(turn N) ~= next.request_received_ms - this.event_time_unix_ms
```

`request_received_ms` is stamped at the frontend before the request enters the router queue or pause path. Server wait time lands in each request's own duration, not in the inter-turn gap. For agentic replay, that gap becomes the inter-request delay. Autodetect cannot split tool execution from agent overhead; it gives the wall-clock union of any parallel tool calls.

<details>
<summary>Optional explicit tool events over ZMQ</summary>

Set `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` to bind the ingress, then have the harness publish tool events. Use explicit tool events when you need per-tool attribution: `duration_ms`, `status`, output size, or error type. Nothing emits tool events on its own.

Wire format is `[topic, seq_be_u64, msgpack(RequestTraceRecord)]`; the default topic is `agent-tool-events`. Use a background publisher, bounded queue, monotonic sequence, and PUSH with HWM. Terminal `tool_end` and `tool_error` rows should carry timing (`started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`) even if `tool_start` was dropped.

Use the same trajectory identity as the surrounding LLM calls. `tool_call_id` should be unique per trajectory. Join offline on `trajectory_id` and `tool_call_id`.

Example `tool_end`:

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "event_source": "harness",
  "agent_context": {
    "trajectory_id": "research-run-42:researcher"
  },
  "tool": {
    "tool_call_id": "call-abc",
    "tool_class": "web_search",
    "status": "succeeded",
    "started_at_unix_ms": 1777312801080,
    "ended_at_unix_ms": 1777312801500,
    "duration_ms": 420.5
  }
}
```

Optional `tool` keys: `output_tokens`, `output_bytes`, `tool_name_hash`, `error_type`. Status values: `running`, `succeeded`, `error`, `cancelled`; synonyms `ok`/`success`, `failed`, `timeout`, and `canceled` also deserialize.

</details>

## Audit Payloads

Request traces do not save input or output payloads by default. To view payloads, enable Dynamo audit sinks next to request tracing.

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_SINKS=jsonl_gz
export DYN_REQUEST_TRACE_OUTPUT_PATH=/tmp/dynamo-trace
export DYN_AUDIT_SINKS=jsonl_gz
export DYN_AUDIT_OUTPUT_PATH=/tmp/dynamo-audit
export DYN_AUDIT_FORCE_LOGGING=true
```

After the run, correlate trace and audit records by request ID:

```bash
gzip -cd /tmp/dynamo-audit.*.jsonl.gz | jq -c '.event' > /tmp/audit.jsonl
gzip -cd /tmp/dynamo-trace.*.jsonl.gz | jq -c '.event // .' > /tmp/trace.jsonl
jq -s 'group_by(.request_id // .request.request_id)' /tmp/audit.jsonl /tmp/trace.jsonl
```

Each JSONL line wraps the record:

```json
{
  "timestamp": 1234,
  "event": { "schema": "dynamo.request.trace.v1", "...": "..." }
}
```

`timestamp` is sink-relative elapsed time in milliseconds. Use `event.event_time_unix_ms` for wall-clock ordering.

## View Traces in Perfetto

Convert request trace JSONL files into a [Perfetto](https://ui.perfetto.dev/) trace file:

```bash
uv run --no-project python benchmarks/request_trace/convert_to_perfetto.py \
  "${DYN_REQUEST_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output "${DYN_REQUEST_TRACE_OUTPUT_PATH}.perfetto.json"
```

Open the output in [Perfetto UI](https://ui.perfetto.dev/). Useful flags include `--include-markers` and `--no-stages`.

Request slices include flattened finish metadata when present, such as `finish.finish_reason`, `finish.backend_finish_reason`, `finish.stop_reason`, `finish.tool_call_count`, `finish.tool_call_names`, and per-choice summaries like `finish.choice_finish_reasons`.

## Replay Agentic Request Traces

**Experimental.** Convert a collected request trace into an agentic Mooncake trace and replay it with `python -m dynamo.replay`. The converter uses Dynamo `request_end` rows for request timing, token lengths, worker placement, and replay hashes. It also uses terminal harness tool rows (`tool_end` / `tool_error`) to preserve tool-wait time between dependent LLM requests.

Replay ignores non-replay request fields such as `finish_reason_metadata`; use the Perfetto view when you want to inspect final finish reasons, backend stop signals, or complete tool-call metadata inside the trace.

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --agentic \
  --input-path "${DYN_REQUEST_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output-file /tmp/dynamo-request-trace.agentic-mooncake.jsonl
```

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

Agentic Mooncake rows preserve:

- `request_id`: the LLM request row identity.
- Mooncake `session_id`: derived from the Dynamo `trajectory_id`.
- `wait_for`: request IDs that must complete before this row becomes eligible.
- `branches`: child request IDs spawned from this row.
- `prefix_reset`: first request in a trajectory.
- `delay`: non-tool delay after dependencies finish.
- `tool_wait_ms`: tool time after dependencies finish, parallel-aware as the union of overlapping spans rather than their sum.
- `tool_events`: per-tool spans attributed to this LLM request, each carrying `tool_call_id`, `tool_class`, `status`, `started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`, and optional `output_bytes`, `output_tokens`, or `error_type`.
- `hash_ids`, `input_length`, and `output_length`: prompt-prefix and length data for mocker replay.

Rows with no `wait_for` use their `timestamp` as the replay start time. Rows with dependencies wait for all listed requests to complete, then wait `delay + tool_wait_ms` before dispatch. For more flags and engine settings, see [DynoSim Runs](../dynosim/runs.md).

<details>
<summary>ATIF alignment</summary>

Dynamo emits `dynamo.request.trace.v1`, not full ATIF logs, but identifiers match [ATIF][atif-rfc] and [Harbor](https://github.com/harbor-framework/harbor) so you can join harness trajectories to Dynamo rows on `trajectory_id`. Dynamo omits conversational payload by design.

| Dynamo | Role |
|--------|------|
| `trajectory_id` | Branch within run |
| `parent_trajectory_id` | Subagent link |

</details>

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
