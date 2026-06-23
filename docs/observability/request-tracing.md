---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Replay Tracing
subtitle: Capture live chat and completion traffic for Mooncake replay
---

Request replay tracing records one `request_end` row for each eligible Rust OpenAI
chat or completion request. Without trajectory headers, the row stays compact
and contains only replay metadata. With trajectory headers, the same
`dynamo.request.trace.v1` stream also includes trajectory identity, request
metrics, finish metadata, and optional harness tool events.

Request tracing does not record prompts, responses, or tool arguments. Use the
audit sink when payload capture is required.

Enable the default rotating gzip sink:

```bash
export DYN_REQUEST_TRACE=1
```

This writes `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`. To choose another
segment prefix:

```bash
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_OUTPUT_PATH=/mnt/captures/run-42/request-trace
```

## Configuration

| Variable | Default when enabled | Description |
| --- | --- | --- |
| `DYN_REQUEST_TRACE` | unset | Truthy master switch. |
| `DYN_REQUEST_TRACE_SINKS` | `jsonl_gz` | Comma-separated `jsonl`, `jsonl_gz`, or `stderr`. |
| `DYN_REQUEST_TRACE_OUTPUT_PATH` | `/tmp/dynamo-request-trace` | Literal JSONL path or gzip segment prefix. |
| `DYN_REQUEST_TRACE_CAPACITY` | `1024` | Best-effort in-process broadcast capacity. |
| `DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES` | `1048576` | JSONL or gzip batching threshold. |
| `DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS` | `1000` | Periodic flush interval. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES` | `268435456` | Gzip roll threshold in uncompressed bytes. |
| `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES` | unset | Optional gzip roll threshold in records. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | Optional ZMQ PULL bind address for harness tool events. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` | `agent-tool-events` | First-frame ZMQ topic filter when endpoint is configured. |

Set the ZMQ endpoint on the process that should own tool-event ingress, usually
the frontend process. If the same bind address is exported to multiple Dynamo
processes, the first process binds it and later processes warn and continue.
The harness should publish `agent-tool-events` as the first ZMQ frame unless
`DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` is set on Dynamo.

The bus and sinks use best-effort delivery behavior.
A slow sink can report lag and drop records. Validate captured row counts before
using a trace as a complete workload.

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
    "trajectory_id": "research-run-42:researcher",
    "parent_trajectory_id": "research-run-42:planner"
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

Optional harness tool events use the `RequestTraceToolEventIngress` payload below. Dynamo normalizes these events into request trace rows before writing them to sinks.

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "trajectory_id": "research-run-42:researcher",
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

`input_sequence_hashes` are Dynamo's sequence-aware rolling hashes. The
Mooncake converter maps them to compact `hash_ids`; they are not copied
verbatim into the Mooncake output.

For a canceled response stream, `output_tokens` is the final partial OSL
observed after the inner response stream has been dropped.

## Supported Requests

Initial coverage is the Rust OpenAI chat-completions and completions paths.
Context-free replay rows must represent one Mooncake request, so tracing skips:

- `n > 1`
- `best_of > 1`
- `prompt_embeds`
- Multimodal inputs
- Requests without a tracker or usable KV cache block size

Skipped requests produce a structured warning and no partial replay row.
Header-derived trajectory context enriches supported request trace rows; it does not bypass
these shape checks or create an agent-only fallback row.

## Convert And Replay

The converter accepts `dynamo.request.trace.v1` captures:

```bash
cargo run -p dynamo-bench --bin request_trace_to_mooncake -- \
  --input-path /tmp/dynamo-request-trace.*.jsonl.gz \
  --output-file /tmp/dynamo-request-trace.mooncake.jsonl
```

Pass `--agentic` only when every request row has `agent_context`. Context-free
request traces still convert to ordinary Mooncake rows and are rejected with
`--agentic`. The converter rejects unknown schema versions.

Use the trace block size printed by the converter for both trace parsing and
the mock engine:

```bash
TRACE_BLOCK_SIZE=64
.venv/bin/python -m dynamo.replay /tmp/dynamo-request-trace.mooncake.jsonl \
  --trace-format mooncake \
  --trace-block-size "${TRACE_BLOCK_SIZE}" \
  --replay-mode offline \
  --router-mode kv_router \
  --num-workers 4 \
  --extra-engine-args "{\"block_size\":${TRACE_BLOCK_SIZE}}" \
  --report-json /tmp/dynamo-request-trace.replay-report.json
```

`DYN_REQUEST_TRACE` is the switch for replay and agent-aware capture. Agent
context does not require a separate trace flag; if trajectory headers are
present, the request trace row is enriched automatically.
