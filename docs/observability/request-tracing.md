---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Replay Tracing
subtitle: Capture live chat and completion traffic for direct DynoSim replay
---

Request replay tracing records `dynamo.request.trace.v1` rows for eligible Rust
OpenAI chat or completion requests. The compact `request_end` row contains
replay metadata. With session headers, the same stream also includes session
identity, request metrics, finish metadata, and optional harness tool events.

Session identity enriches traces only. Its presence does not enable sticky sessions or change routing policy.

Request trace can also emit `request_payload` rows for OpenAI
`/v1/chat/completions` requests. Payload rows include the client request and,
when the response completes, the response. By default, Dynamo does not emit
request or response payload rows, even when `store=true`. Set
`DYN_REQUEST_TRACE_INCLUDE_REQUEST_RESPONSE=true` to emit payload rows for every
eligible chat request.

Enable the default rotating gzip file destination:

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
| `DYN_REQUEST_TRACE` | unset | Truthy value | Master switch. |
| `DYN_REQUEST_TRACE_DESTINATIONS` | `file` | `file`, `stderr`, `nats`, `otel` | Comma-separated record destinations. |
| `DYN_REQUEST_TRACE_FILE_PATH` | `/tmp/dynamo-request-trace` | File path or segment prefix | Literal path when compression is `none`; gzip segment prefix when compression is `gzip`. |
| `DYN_REQUEST_TRACE_FILE_FORMAT` | `jsonl` | `jsonl` | File record format. |
| `DYN_REQUEST_TRACE_FILE_COMPRESSION` | `gzip` | `gzip`, `none` | File compression. `gzip` writes `<prefix>.<index>.jsonl.gz`; `none` writes a literal JSONL path. |
| `DYN_REQUEST_TRACE_CAPACITY` | `1024` | Positive integer | Best-effort in-process broadcast capacity. |
| `DYN_REQUEST_TRACE_INCLUDE_REQUEST_RESPONSE` | `false` | `true`, `false` | Include request and response payload bodies by emitting `request_payload` rows for all eligible chat requests. When `false`, no payload rows are emitted, even if `store=true`. |
| `DYN_REQUEST_TRACE_NATS_SUBJECT` | `dynamo.request_trace.v1` | NATS subject | Subject used when `DYN_REQUEST_TRACE_DESTINATIONS` includes `nats`. |
| `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` | `4194304` | Positive integer bytes | Max serialized OTEL payload attribute size. Oversized `request_payload` rows emit a marker with `payload_complete=false` and `payload_drop_reason`. |
| `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | `1048576` | Integer bytes | File batching threshold. |
| `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` | `1000` | Integer milliseconds | Periodic flush interval. |
| `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | `268435456` | Positive integer bytes | Gzip roll threshold in uncompressed bytes. |
| `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | unset | Positive integer records | Optional gzip roll threshold in records. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | ZMQ bind address | Optional ZMQ PULL bind address for harness tool events. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` | `agent-tool-events` | ZMQ topic | First-frame ZMQ topic filter when endpoint is configured. |

## Compatibility and Deprecated Aliases

> [!WARNING]
> Deprecated. The aliases in this section remain accepted for compatibility.
> Prefer canonical `DYN_REQUEST_TRACE_*` variables for new deployments.

Deprecated request trace and audit variables remain accepted as fallbacks. New
request trace variables take precedence over request trace aliases, and request
trace aliases take precedence over audit aliases.

Destination selection resolves in this order:

1. `DYN_REQUEST_TRACE_DESTINATIONS`
2. `DYN_REQUEST_TRACE_SINKS`
3. `DYN_AUDIT_SINKS`
4. `file` when request trace is enabled and no destination variable is set

`DYN_AUDIT_SINKS` also enables request trace compatibility when
`DYN_REQUEST_TRACE` is unset. Other deprecated aliases configure an enabled
request trace, but do not enable request trace by themselves.

### Changed Request Trace Variables

| Previous variable | Canonical variable | Still respected? | Notes |
| --- | --- | --- | --- |
| `DYN_REQUEST_TRACE_SINKS` | `DYN_REQUEST_TRACE_DESTINATIONS` | Yes, when `DYN_REQUEST_TRACE_DESTINATIONS` is unset. | Prefer `file`, `stderr`, `nats`, or `otel`. Legacy `jsonl` maps to `file` with `DYN_REQUEST_TRACE_FILE_COMPRESSION=none`; `jsonl_gz` maps to `file` with `DYN_REQUEST_TRACE_FILE_COMPRESSION=gzip`. |
| `DYN_REQUEST_TRACE_OUTPUT_PATH` | `DYN_REQUEST_TRACE_FILE_PATH` | Yes, when `DYN_REQUEST_TRACE_FILE_PATH` is unset. | Used only when the `file` destination is configured. |
| `DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES` | `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | Yes, when `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` is unset. | Applies to `file` destination buffering. |
| `DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS` | `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` | Yes, when `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` is unset. | Applies to `file` destination flushing. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES` | `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | Yes, when `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` is unset. | Applies when `DYN_REQUEST_TRACE_FILE_COMPRESSION=gzip`. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES` | `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | Yes, when `DYN_REQUEST_TRACE_FILE_ROLL_LINES` is unset. | Applies when `DYN_REQUEST_TRACE_FILE_COMPRESSION=gzip`. |

### Added Request Trace Variables

| Variable | Values | Purpose | Fallback behavior |
| --- | --- | --- | --- |
| `DYN_REQUEST_TRACE_DESTINATIONS` | `file`, `stderr`, `nats`, `otel` | Selects record destinations. | Falls back to `DYN_REQUEST_TRACE_SINKS`, then `DYN_AUDIT_SINKS`, then `file` when request trace is enabled. |
| `DYN_REQUEST_TRACE_FILE_PATH` | File path or segment prefix | Configures the `file` destination path. | Falls back to `DYN_REQUEST_TRACE_OUTPUT_PATH`, then `DYN_AUDIT_OUTPUT_PATH`, then `/tmp/dynamo-request-trace` when the `file` destination is enabled. |
| `DYN_REQUEST_TRACE_FILE_FORMAT` | `jsonl` | Selects the file record format. | Defaults to `jsonl`; unknown values warn and use `jsonl`. |
| `DYN_REQUEST_TRACE_FILE_COMPRESSION` | `gzip`, `gz`, `jsonl_gz`, `none`, `off`, `false` | Selects file compression. | Falls back to compression inferred from legacy `jsonl` or `jsonl_gz` destination values, then `gzip`. |
| `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | Integer bytes | Sets file destination buffer size. | Falls back to `DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES`, then `DYN_AUDIT_JSONL_BUFFER_BYTES`, then `1048576`. |
| `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` | Integer milliseconds | Sets file destination flush interval. | Falls back to `DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS`, then `DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS`, then `1000`. |
| `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | Positive integer bytes | Sets gzip roll threshold in uncompressed bytes. | Falls back to `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES`, then `DYN_AUDIT_JSONL_GZ_ROLL_BYTES`, then `268435456`. |
| `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | Positive integer records | Sets optional gzip roll threshold in records. | Falls back to `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES`, then `DYN_AUDIT_JSONL_GZ_ROLL_LINES`; unset means no line-count roll threshold. |
| `DYN_REQUEST_TRACE_INCLUDE_REQUEST_RESPONSE` | `true`, `false` | Emits `request_payload` rows with request and response bodies for eligible chat requests. | Falls back to `DYN_AUDIT_FORCE_LOGGING`, then `false`. When `false`, no payload rows are emitted, even if `store=true`. |
| `DYN_REQUEST_TRACE_NATS_SUBJECT` | NATS subject | Sets the `nats` destination subject. | Falls back to `DYN_AUDIT_NATS_SUBJECT`, then `dynamo.request_trace.v1`. |
| `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` | Positive integer bytes | Sets the max serialized OTEL payload attribute size. | Falls back to `DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES`, then `4194304`. |

### Deprecated Audit Aliases

| Deprecated variable | Canonical variable | Still respected? | Notes |
| --- | --- | --- | --- |
| `DYN_AUDIT_SINKS` | `DYN_REQUEST_TRACE_DESTINATIONS` | Yes, when neither `DYN_REQUEST_TRACE_DESTINATIONS` nor `DYN_REQUEST_TRACE_SINKS` is set. | Also enables request trace compatibility when `DYN_REQUEST_TRACE` is unset. Legacy `jsonl` and `jsonl_gz` map to the `file` destination and set compression as described above. |
| `DYN_AUDIT_FORCE_LOGGING` | `DYN_REQUEST_TRACE_INCLUDE_REQUEST_RESPONSE` | Yes, when `DYN_REQUEST_TRACE_INCLUDE_REQUEST_RESPONSE` is unset. | Preserves the legacy opt-in payload logging behavior under the new request trace record schema. |
| `DYN_AUDIT_CAPACITY` | `DYN_REQUEST_TRACE_CAPACITY` | Yes, when `DYN_REQUEST_TRACE_CAPACITY` is unset. | Configures request trace bus capacity. |
| `DYN_AUDIT_NATS_SUBJECT` | `DYN_REQUEST_TRACE_NATS_SUBJECT` | Yes, when `DYN_REQUEST_TRACE_NATS_SUBJECT` is unset. | Configures the `nats` destination subject. |
| `DYN_AUDIT_OUTPUT_PATH` | `DYN_REQUEST_TRACE_FILE_PATH` | Yes, when `DYN_REQUEST_TRACE_FILE_PATH` and `DYN_REQUEST_TRACE_OUTPUT_PATH` are unset. | Used only for the `file` destination. |
| `DYN_AUDIT_JSONL_BUFFER_BYTES` | `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES` | Yes, when request trace buffer env vars are unset. | Configures file destination buffering. |
| `DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS` | `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS` | Yes, when request trace flush env vars are unset. | Configures file destination flushing. |
| `DYN_AUDIT_JSONL_GZ_ROLL_BYTES` | `DYN_REQUEST_TRACE_FILE_ROLL_BYTES` | Yes, when request trace roll-byte env vars are unset. | Applies when file compression is `gzip`. |
| `DYN_AUDIT_JSONL_GZ_ROLL_LINES` | `DYN_REQUEST_TRACE_FILE_ROLL_LINES` | Yes, when request trace roll-line env vars are unset. | Applies when file compression is `gzip`. |
| `DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES` | `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` | Yes, when `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES` is unset. | Configures the `otel` destination payload size guard. |

Set the ZMQ endpoint on the process that should own tool-event ingress, usually
the frontend process. If the same bind address is exported to multiple Dynamo
processes, the first process binds it and later processes warn and continue.
The harness should publish `agent-tool-events` as the first ZMQ frame unless
`DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` is set on Dynamo.

The bus and destinations use best-effort delivery behavior.
A slow destination can report lag and drop records. Validate captured row counts before
using a trace as a complete workload.

The `otel` destination uses the standard `OTEL_EXPORTER_OTLP_*` variables. Set
`OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` and `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` to
route request trace records through an OpenTelemetry Collector. The `otel`
destination writes each request trace row as one OTLP log record with the full
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
    "requested_streaming": true,
    "model": "my-model",
    "request": {
      "model": "my-model",
      "messages": [
        {
          "role": "user",
          "content": "Hello"
        }
      ],
      "store": true
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
destination drops an oversized payload body, the row contains
`payload_complete=false` and `payload_drop_reason`.

Optional harness tool events use the `RequestTraceToolEventIngress` payload below. Dynamo normalizes these events into request trace rows before writing them to destinations.

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
