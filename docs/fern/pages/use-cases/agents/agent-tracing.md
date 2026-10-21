---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Tracing
subtitle: Capture request traces, tool timing, and Perfetto timelines for agent runs
---

By the end of this guide, you have a trace of an agent run and a Perfetto timeline that shows each LLM request, its prefill and decode stages, and the time the agent spent in tools.

Agent tracing records request timing, token counts, worker placement, finish metadata, and replay hashes for eligible LLM requests. Requests with [session identity](session-ids.mdx) also carry agent context, which lets analysis tools group LLM turns and tool activity into the same run.

<a id="enable-output"></a>

## Capture Your First Agent Trace

<Steps>

<Step title="Enable request tracing">

Set the master switch before starting the Dynamo frontend:

```bash title="Enable the default gzip trace sink"
export DYN_REQUEST_TRACE=1
```

Dynamo writes rotating gzip-compressed JSONL segments to `/tmp/dynamo-request-trace.NNNNNN.jsonl.gz`.

To use another segment prefix, set `DYN_REQUEST_TRACE_FILE_PATH`:

```bash title="Choose an output path"
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_FILE_PATH=/mnt/captures/run-42/request-trace
```

Default request traces do not include prompts, responses, or tool-call arguments. The optional [`request_payload` record](../../reference/observability/request-traces.mdx#request_payload) does, so enable it only if you need it. Traces do include client-supplied agent headers, so keep secrets out of those headers. See [What a Trace Contains](#what-a-trace-contains).

</Step>

<Step title="Run your agent workload">

Send requests through a configured [agent harness](agent-harnesses.mdx) or a custom client. Session identity enriches eligible `request_end` rows automatically; it does not need another trace flag.

</Step>

<Step title="Convert the capture to Perfetto">

Run the converter from the Dynamo repository. It accepts `.jsonl` and `.jsonl.gz` files, a directory of segments, or a quoted glob:

```bash title="Create a Perfetto trace"
python3 benchmarks/request_trace/convert_to_perfetto.py \
  "${DYN_REQUEST_TRACE_FILE_PATH:-/tmp/dynamo-request-trace}".*.jsonl.gz \
  --output /tmp/run-42.perfetto.json
```

</Step>

<Step title="Open the timeline">

Open the output file in the [Perfetto UI](https://ui.perfetto.dev/). The default view shows one lane per session, LLM request slices with prefill and decode stages, and tool slices when the trace contains tool metadata.

</Step>

</Steps>

For every request-trace environment variable and default, see the [Request Trace Reference](../../reference/observability/request-traces.mdx#configuration).

## Choose the Tool Detail You Need

| Capture method | Harness changes | What it records |
|---|---|---|
| Automatic tool metadata | None | Tool-call names and IDs from response parsing, plus inferred tool-wait spans in Perfetto |
| Explicit harness events | Required | Per-tool timing, status, output size, and error type |

### Record Tool Calls Automatically

When tracing is enabled and the worker uses `--dyn-tool-call-parser`, Dynamo records the final finish reason and each parsed tool call's name and ID in `request_end.finish_reason_metadata`. Tool-call arguments are omitted on purpose.

The Perfetto converter infers a tool span from the end of a tool-call response to the next request in the same session. You can calculate the same interval from the trace:

```text title="Approximate tool and agent wait time"
tool_wait(turn N) ~= next.request_received_ms - this.event_time_unix_ms
```

> [!IMPORTANT]
> This interval includes tool execution and agent overhead. It cannot separate parallel tools or attribute time to one tool.

### Record Precise Tool Timing

For per-tool attribution, have the harness send events to the frontend. Configure the frontend:

```bash title="Enable explicit tool-event ingress"
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390
```

The endpoint is a ZMQ PULL bind address. Configure it on only one process. The harness publishes `agent-tool-events` as the first frame unless you override the topic with `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC`.

<details>
<summary>Tool-event wire format and example</summary>

The wire format is `[topic, seq_be_u64, msgpack(tool_event)]`. `tool_event` has top-level `schema`, `event_type`, `event_time_unix_ms`, `session_id`, an optional `parent_session_id`, and `tool`. Dynamo adds the `harness` event source and builds the record's agent context from the session fields. Use a monotonic sequence and a background PUSH publisher with a bounded queue. Include `started_at_unix_ms`, `ended_at_unix_ms`, and `duration_ms` on terminal `tool_end` and `tool_error` events so timing survives a dropped `tool_start`.

Use the same `session_id` as the surrounding LLM requests. Make each `tool_call_id` unique within its session.

```json title="Explicit tool_end event"
{
  "schema": "dynamo.request.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "session_id": "research-run-42:researcher",
  "parent_session_id": "research-run-42:planner",
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

<a id="dynamo-request_end-record"></a>

## What a Trace Contains

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
    "session_id": "research-run-42:researcher",
    "parent_session_id": "research-run-42:planner"
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

> [!IMPORTANT]
> Default request traces do not capture prompt, response, or tool-call argument bodies. However, `agent_context.agent_headers` contains unredacted, client-supplied values and can include arbitrary data. These headers appear on `request_end` rows independently of `DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST`, which applies only to optional `request_payload` rows. Do not send secrets or prompt content in agent metadata headers. See the [Request Trace Reference](../../reference/observability/request-traces.mdx#request_end) for the capture boundary.

Requests with recognized session identity can include opaque strings in `agent_context.agent_headers`. Trace consumers decide how to interpret them and must tolerate unknown names, values, and embedded JSON. Captured headers do not change request placement or evict cached blocks. See [Captured agent headers](session-ids.mdx#agent-headers) for capture rules and limits. Older frontends cannot supply the map during a rolling upgrade.

For chat streams, Dynamo records finish metadata after parser and jail rewrites. Completion streams record the final OpenAI-compatible completion finish reason.

> [!WARNING]
> Request tracing currently covers eligible Rust OpenAI chat-completions and completions requests. It skips unsupported replay shapes, including `n > 1`, `best_of > 1`, `prompt_embeds`, multimodal inputs, and requests without a tracker or usable KV cache block size. Sinks use best-effort delivery and can drop records when they lag, so check warnings and validate row counts before treating a capture as complete.

## Next

- [Agent simulation](agent-simulation.mdx) turns a capture into a replayable benchmark.
- [Maintain session affinity](session-affinity.md) and [Prioritize requests](priority-scheduling.mdx) are changes you can now measure with a trace.
