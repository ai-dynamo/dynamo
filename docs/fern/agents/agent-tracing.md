---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Export Dynamo request traces, tool-call metadata, and Perfetto timelines
---

Agent tracing captures request timing, token counts, worker placement, finish metadata, and replay hashes for eligible LLM requests. Requests with [session identity](session-ids.mdx) also carry agent context, which lets analysis tools group LLM turns and tool activity into the same run.

<Info>Request traces contain metadata, not payloads. Dynamo does not store prompts, responses, or tool-call arguments in these traces.</Info>

<a id="enable-output"></a>

## Capture Your First Agent Trace

<Steps>

<Step title="Enable request tracing">

Set the master switch before starting the Dynamo frontend:

```bash title="Enable the default gzip trace sink"
export DYN_REQUEST_TRACE=1
```

Dynamo writes rotating gzip-compressed JSONL segments to `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`.

To use another segment prefix, set `DYN_REQUEST_TRACE_OUTPUT_PATH`:

```bash title="Choose an output path"
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_OUTPUT_PATH=/mnt/captures/run-42/request-trace
```

</Step>

<Step title="Run your agent workload">

Send requests through a configured [Agent Harness](agent-harnesses.md) or custom client. Session identity enriches eligible `request_end` rows automatically; it does not require another trace flag.

</Step>

<Step title="Convert the capture to Perfetto">

Run the converter from the Dynamo repository:

```bash title="Create a Perfetto trace"
python3 benchmarks/request_trace/convert_to_perfetto.py \
  "${DYN_REQUEST_TRACE_OUTPUT_PATH}".*.jsonl.gz \
  --output "${DYN_REQUEST_TRACE_OUTPUT_PATH}.perfetto.json"
```

</Step>

<Step title="Open the timeline">

Open `/tmp/dynamo-request-trace.perfetto.json` in the [Perfetto UI](https://ui.perfetto.dev/). The default view shows LLM request slices, prefill and decode stages, and tool slices when the trace contains tool metadata.

</Step>

</Steps>

For every request-trace environment variable and default, see [Request Replay Tracing](../observability/request-tracing.md#configuration).

## Choose the Tool Detail You Need

| Capture method | Harness changes | What it records |
|----------------|-----------------|-----------------|
| Automatic tool metadata | None | Tool-call names and IDs from response parsing, plus inferred tool-wait spans in Perfetto |
| Explicit harness events | Required | Per-tool timing, status, output size, and error type |

## Tool Call Observability

### Record Tool Calls Automatically

When request tracing is enabled and the worker uses `--dyn-tool-call-parser`, Dynamo records the final finish reason and each parsed tool call's name and ID in `request_end.finish_reason_metadata`. Tool-call arguments are intentionally omitted.

The Perfetto converter infers a tool span from the end of a tool-call response to the next request in the same trajectory. You can calculate the same interval from the trace:

```text title="Approximate tool and agent wait time"
tool_wait(turn N) ~= next.request_received_ms - this.event_time_unix_ms
```

<Info>This interval includes tool execution and agent overhead. It cannot separate parallel tools or attribute time to one tool.</Info>

### Record Precise Tool Timing

For per-tool attribution, configure the frontend to receive events from the harness:

```bash title="Enable explicit tool-event ingress"
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390
```

The endpoint is a ZMQ PULL bind address. The harness publishes `agent-tool-events` as the first frame unless you override the topic with `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC`.

<details>
<summary>Tool-event wire format and example</summary>

The wire format is `[topic, seq_be_u64, msgpack(RequestTraceRecord)]`. Use a monotonic sequence and a background PUSH publisher with a bounded queue. Include `started_at_unix_ms`, `ended_at_unix_ms`, and `duration_ms` on terminal `tool_end` and `tool_error` events so timing survives a dropped `tool_start`.

Use the same agent context as the surrounding LLM requests. Make each `tool_call_id` unique within its trajectory.

```json title="Explicit tool_end event"
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "event_source": "harness",
  "agent_context": {
    "session_id": "research-run-42",
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

Optional `tool` fields are `output_tokens`, `output_bytes`, `tool_name_hash`, and `error_type`. Status values are `running`, `succeeded`, `error`, and `cancelled`; supported aliases include `ok`, `success`, `failed`, `timeout`, and `canceled`.

</details>

## Dynamo `request_end` Record

Dynamo emits `request_end` after an eligible response stream finishes or is dropped. The record can contain agent context, request timing and token metrics, worker information, finish metadata, and replay hashes.

<details>
<summary>Full agent-enriched <code>request_end</code> example</summary>

```json
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "session_id": "research-run-42",
    "trajectory_id": "research-run-42:researcher",
    "parent_trajectory_id": "research-run-42:planner"
  },
  "request": {
    "request_id": "dynamo-request-id",
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

</details>

For chat streams, Dynamo records finish metadata after parser and jail rewrites. Completion streams record the final OpenAI-compatible completion finish reason.

<Warning>Request tracing currently covers eligible Rust OpenAI chat-completions and completions requests. It skips unsupported replay shapes, including `n > 1`, `best_of > 1`, `prompt_embeds`, multimodal inputs, and requests without a tracker or usable KV cache block size. Sinks use best-effort delivery and can drop records when they lag, so check warnings and validate row counts before treating a capture as complete.</Warning>


