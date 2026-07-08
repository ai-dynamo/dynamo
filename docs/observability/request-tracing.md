---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Replay Tracing
subtitle: Capture live chat and completion traffic for direct DynoSim replay
---

Request replay tracing records `dynamo.request.trace.v1` rows for eligible
OpenAI chat or completion requests. The compact `request_end` row contains
replay metadata. With session headers, the same stream also includes session
identity, request metrics, finish metadata, and optional harness tool events.

Session identity enriches traces only. Its presence does not enable sticky sessions or change routing policy.

Request trace can also emit `request_payload` rows for OpenAI
`/v1/chat/completions` requests. Payload rows include the client request and,
when the response completes, the response. By default, Dynamo does not emit
request or response payload rows, even when `store=true`. Set
`DYN_REQUEST_TRACE_RECORDS=request_payload` to emit payload rows for every
eligible chat request, or include `request_payload` alongside other selected
record types.

Enable the default rotating gzip file sink:

```bash
export DYN_REQUEST_TRACE=1
```

This writes `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`. To choose another
segment prefix:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_FILE_PATH=/mnt/captures/run-42/request-trace
```

## Configuration

| Variable | Default when enabled | Values | Description |
| --- | --- | --- | --- |
| `DYN_REQUEST_TRACE` | unset | Truthy value | Master switch. When enabled and `DYN_REQUEST_TRACE_RECORDS` is unset, emits `request_end,tool`. |
| `DYN_REQUEST_TRACE_RECORDS` | `request_end,tool` when `DYN_REQUEST_TRACE=1`; unset otherwise | `request_end`, `request_payload`, `tool` | Comma-separated record types to emit. Setting this variable enables only the listed records. |
| `DYN_REQUEST_TRACE_SINKS` | `file` | `file`, `stderr`, `nats`, `otel` | Comma-separated record sinks. |
| `DYN_REQUEST_TRACE_FILE_PATH` | `/tmp/dynamo-request-trace` | File path or segment prefix | Literal path when `DYN_REQUEST_TRACE_FILE_FORMAT=jsonl`; gzip segment prefix when `DYN_REQUEST_TRACE_FILE_FORMAT=jsonl_gz`. |
| `DYN_REQUEST_TRACE_FILE_FORMAT` | `jsonl_gz` | `jsonl`, `jsonl_gz` | File record format. `jsonl_gz` writes `<prefix>.<index>.jsonl.gz`; `jsonl` writes a literal JSONL path. |
| `DYN_REQUEST_TRACE_CAPACITY` | `1024` | Positive integer | Best-effort in-process broadcast capacity. |
| `DYN_REQUEST_TRACE_NATS_SUBJECT` | `dynamo.request_trace.v1` | NATS subject | Subject used when `DYN_REQUEST_TRACE_SINKS` includes `nats`. |
| `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` | `4194304` | Positive integer bytes | Max serialized OTEL payload attribute size. Oversized `request_payload` rows emit a marker with `payload_complete=false` and `payload_drop_reason`. |
| `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | `1048576` | Integer bytes | File batching threshold. |
| `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` | `1000` | Integer milliseconds | Periodic flush interval. |
| `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | `268435456` | Positive integer bytes | Gzip roll threshold in uncompressed bytes. |
| `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | unset | Positive integer records | Optional gzip roll threshold in records. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | ZMQ bind address | Optional ZMQ PULL bind address for harness tool events. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` | `agent-tool-events` | ZMQ topic | First-frame ZMQ topic filter when endpoint is configured. |
| `DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST` | unset (none) | Comma/whitespace-separated header names | Allowlist of HTTP request header names to record in `request_payload` rows (`payload.http_request_headers`), case-insensitive. Only listed headers are captured; unset/empty captures none. Applies to every sink. |

> [!WARNING]
> Deprecated. The legacy `jsonl` and `jsonl_gz` values for
> `DYN_REQUEST_TRACE_SINKS`, `DYN_REQUEST_TRACE_OUTPUT_PATH`, and
> `DYN_REQUEST_TRACE_JSONL_*` aliases remain accepted for compatibility.
> `DYN_REQUEST_TRACE_SINKS=jsonl` maps to the `file` sink with
> `DYN_REQUEST_TRACE_FILE_FORMAT=jsonl`; `jsonl_gz` maps to the `file` sink
> with `DYN_REQUEST_TRACE_FILE_FORMAT=jsonl_gz`. The legacy audit variables
> `DYN_AUDIT_SINKS`, `DYN_AUDIT_FORCE_LOGGING`, `DYN_AUDIT_OUTPUT_PATH`,
> `DYN_AUDIT_NATS_SUBJECT`, `DYN_AUDIT_JSONL_*`, and
> `DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES` are accepted as migration shims, not
> legacy wire-compatibility aliases. A truthy `DYN_AUDIT_FORCE_LOGGING` maps to
> `DYN_REQUEST_TRACE_RECORDS=request_payload`; `DYN_AUDIT_SINKS` only selects
> destinations and does not enable `request_end` replay metadata.

Set the ZMQ endpoint on the process that should own tool-event ingress, usually
the frontend process. If the same bind address is exported to multiple Dynamo
processes, the first process binds it and later processes warn and continue.
The harness should publish `agent-tool-events` as the first ZMQ frame unless
`DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` is set on Dynamo.

The bus and sinks use best-effort delivery behavior.
A slow sink can report lag and drop records. Validate captured row counts before
using a trace as a complete workload.

The `otel` sink uses the standard `OTEL_EXPORTER_OTLP_*` variables. Set
`OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` and `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` to
route request trace records through an OpenTelemetry Collector. The `otel`
sink writes each request trace row as one OTLP log record with the full
row serialized in the `payload` attribute.

## Record Shape

Context-free replay row:

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "request": {
    "request_id": "dynamo-request-id",
    "request_received_ms": 1777312800000,
    "output_tokens": 16,
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [
        14879255164371896291,
        274632075616497421
      ]
    }
  }
}
```

Agent-enriched row:

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
    "x_request_id": "caller-request-id",
    "model": "my-model",
    "input_tokens": 128,
    "output_tokens": 16,
    "request_received_ms": 1777312800000,
    "total_time_ms": 1000,
    "finish_reason_metadata": {
      "finish_reason": "tool_calls",
      "tool_calls": [
        {
          "choice_index": 0,
          "tool_call_index": 0,
          "id": "call-abc",
          "name": "web_search"
        }
      ]
    },
    "replay": {
      "trace_block_size": 64,
      "input_length": 128,
      "input_sequence_hashes": [
        14879255164371896291,
        274632075616497421
      ]
    }
  }
}
```

Payload row:

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_payload",
  "event_time_unix_ms": 1777312800000,
  "event_source": "dynamo",
  "payload": {
    "request_id": "dynamo-request-id",
    "endpoint": "openai.chat_completion",
    "model": "my-model",
    "request": {
      "model": "my-model",
      "messages": [
        {
          "role": "user",
          "content": "Hello"
        }
      ],
    },
    "response": {
      "id": "chatcmpl-example",
      "object": "chat.completion",
      "created": 1777312801,
      "model": "my-model",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "payload_complete": true
  }
}
```

For canceled streams, gateway timeouts, and aggregation failures, the row still
contains `payload.request`; `payload.response` is omitted. If the `otel`
sink drops an oversized payload body, the row contains
`payload_complete=false` and `payload_drop_reason`.

Optional harness tool events use the `RequestTraceToolEventIngress` payload below. Dynamo normalizes these events into request trace rows before writing them to sinks.

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

`input_sequence_hashes` are Dynamo's sequence-aware rolling hashes. Replay maps
them to compact internal IDs while loading the original request trace; it does
not write an intermediate Mooncake file.

For a canceled response stream, `output_tokens` is the final partial OSL
observed after the inner response stream has been dropped.

## Supported Requests

Initial coverage is the Rust OpenAI chat-completions and completions paths.
Each context-free replay row must represent one model request, so tracing skips:

- `n > 1`
- `best_of > 1`
- `prompt_embeds`
- Multimodal inputs
- Requests without a tracker or usable KV cache block size

Skipped requests produce a structured warning and no partial replay row.
Header-derived session context enriches supported request trace rows; it does not bypass
these shape checks or create an agent-only fallback row.

## Replay Request Traces

Pass `dynamo.request.trace.v1` JSONL or JSONL.GZ shards directly to replay:

```bash
python -m dynamo.replay /tmp/dynamo-request-trace.*.jsonl.gz \
  --trace-format dynamo \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

No format conversion or intermediate Mooncake file is required.

Replay derives and validates the trace block size across all shards.
Context-free rows use standard replay. If every request has `agent_context`,
replay preserves session dependencies and tool waits. Mixed traces are
rejected.

`DYN_REQUEST_TRACE` is the switch for replay and agent-aware capture. Agent
context does not require a separate trace flag; if session headers are
present, the request trace row is enriched automatically.
