---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Export Dynamo request traces, tool-call metadata, and Perfetto timelines
---

Agent tracing records what Dynamo measured for each eligible LLM request. When a request carries [session identity](session-ids.md), trace rows include the session fields so you can join LLM requests, inferred tool calls, optional harness tool spans, and Perfetto slices. Recording session identity does not enable sticky sessions or session-aware routing.

Tracing is best-effort profiling data, not a compliance audit log. Dynamo does not store tool-call arguments in request traces. Include `request_payload` in `DYN_REQUEST_TRACE_RECORDS` when you need request or response payloads.

## Enable Output

The fast path is one environment variable:

```bash
export DYN_REQUEST_TRACE=1
```

That selects gzip-compressed JSONL file output at `/tmp/dynamo-request-trace.*.jsonl.gz`. Tool-call understanding works immediately from `request_end` finish metadata: no harness tooling required. The optional ZMQ tool-event ingress is opt-in; see [Tool Call Observability](#tool-call-observability).

To relocate captures, set an output path:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_FILE_PATH=/mnt/captures/run-42/request-trace
```

`DYN_REQUEST_TRACE` is the only trace switch. The same request trace stream contains compact replay rows when no session identity is present and enriched agent rows when it is. All request trace variables are documented in [Request Replay Tracing](../observability/request-tracing.md).

## Dynamo `request_end` Record

Dynamo emits `request_end` after the response stream finishes or is dropped. The record carries session identity, `output_tokens`, and autodetected `finish_reason_metadata` such as tool-call names and finish reasons. `request_id` correlates with `request_payload` rows when payload logging is enabled. The `replay` block lets DynoSim load the original request trace directly when Dynamo can represent the request as one replay request. Tool-call metadata is IDs and names only; arguments are intentionally not stored.

<details>
<summary>Full <code>request_end</code> record</summary>

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "session_id": "research-run-42:researcher",
    "parent_session_id": "research-run-42:planner"
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

Current request tracing skips unsupported multi-choice replay shapes such as `n > 1` and `best_of > 1`, so do not assume every session turn is present unless skipped-row warnings are absent. For chat streams, finish metadata is recorded after parser and jail rewrites. Completion streams record the final OpenAI-compatible completion finish reason.

</details>

## Tool Call Observability

Default behavior requires no harness work. Dynamo parses each response stream and records the tool calls the model made into [`request_end.finish_reason_metadata`](#dynamo-request_end-record): the per-turn `finish_reason` and each call's `name` and `id`. Arguments are never stored. This is active whenever `DYN_REQUEST_TRACE=1` and the worker runs a tool-call parser with `--dyn-tool-call-parser`.

You can recover tool-wait time offline without tool events. Within a session, the agent is sequential, so the gap between one turn finishing and the next arriving is the tool plus agent-overhead time:

```text
tool_wait(turn N) ~= next.request_received_ms - this.event_time_unix_ms
```

`request_received_ms` is stamped at the frontend before the request enters the router queue or pause path. Server wait time lands in each request's own duration, not in the inter-turn gap. For agentic replay, that gap becomes the inter-request delay. Autodetect cannot split tool execution from agent overhead; it gives the wall-clock union of any parallel tool calls.

<details>
<summary>Optional explicit tool events over ZMQ</summary>

For precise tool call timing information, you can have your agent harness send tool call events with the relevant `session_id` attached. Set `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` to bind the ingress, then have the harness publish tool events. Use this when you need per-tool attribution: `duration_ms`, `status`, output size, or error type.

Wire format is `[topic, seq_be_u64, msgpack(RequestTraceToolEventIngress)]`; the default topic is `agent-tool-events`. Use a background publisher, bounded queue, monotonic sequence, and PUSH with HWM. Terminal `tool_end` and `tool_error` rows should carry timing (`started_at_unix_ms`, `ended_at_unix_ms`, `duration_ms`) even if `tool_start` was dropped.

Use the same session identity as the surrounding LLM calls. Dynamo converts `session_id` and `parent_session_id` into the internal request trace context. `tool_call_id` should be unique per session. Join offline on `session_id` and `tool_call_id`.

Example `tool_end`:

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "session_id": "research-run-42:researcher",
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

Optional top-level key: `parent_session_id`. Optional `tool` keys: `output_tokens`, `output_bytes`, `tool_name_hash`, `error_type`. Status values: `running`, `succeeded`, `error`, `cancelled`; synonyms `ok`/`success`, `failed`, `timeout`, and `canceled` also deserialize.

</details>

## Request Payloads

Request traces do not save input or output payloads unless payload logging is
enabled. To include chat-completion payload rows in the same request trace
stream, select both `request_end` and `request_payload` records with
`DYN_REQUEST_TRACE_RECORDS=request_end,request_payload`.

```bash
export DYN_REQUEST_TRACE_RECORDS=request_end,request_payload
export DYN_REQUEST_TRACE_SINKS=file
export DYN_REQUEST_TRACE_FILE_PATH=/tmp/dynamo-trace
export DYN_REQUEST_TRACE_FILE_COMPRESSION=gzip
```

After the run, split metadata and payload rows by `event_type`:

```bash
gzip -cd /tmp/dynamo-trace.*.jsonl.gz | jq -c '.event // .' > /tmp/trace.jsonl
jq -c 'select(.event_type == "request_end")' /tmp/trace.jsonl > /tmp/request-end.jsonl
jq -c 'select(.event_type == "request_payload")' /tmp/trace.jsonl > /tmp/request-payload.jsonl
```

Each JSONL line wraps the record:

```json
{
  "timestamp": 1234,
  "event": { "schema": "dynamo.request.trace.v1", "...": "..." }
}
```

`timestamp` is sink-relative elapsed time in milliseconds. Use
`event.event_time_unix_ms` for wall-clock ordering.

## View Traces in Perfetto

Convert request trace JSONL files into a [Perfetto](https://ui.perfetto.dev/) trace file:

```bash
uv run --no-project python benchmarks/request_trace/convert_to_perfetto.py \
  "${DYN_REQUEST_TRACE_FILE_PATH}".*.jsonl.gz \
  --output "${DYN_REQUEST_TRACE_FILE_PATH}.perfetto.json"
```

Open the output in [Perfetto UI](https://ui.perfetto.dev/). The default view shows the normal request stack for LLM requests, backend stages, and tool spans when present.

To replay collected traces using the dynamo mock inference engines, see [Agent Simulation](agent-replay.md).
