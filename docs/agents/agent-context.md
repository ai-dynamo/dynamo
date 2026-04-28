---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Context and Tracing
subtitle: Attach workflow and program identity to agentic requests
---

Agent context is behavior-neutral identity metadata for agentic workloads. It
lets an agent harness tell Dynamo which workflow and reason/tool trajectory a
request belongs to without changing routing, scheduling, or cache behavior.

When agent tracing is enabled, Dynamo joins this identity with request-side
metrics such as token counts, timing, cache hits, queue depth, and worker IDs.
Harnesses can also publish tool lifecycle records into Dynamo's event plane.
Dynamo normalizes both sources onto one trace bus, with JSONL as an optional
local sink.

## Request Contract

Set `nvext.agent_context` on chat completion requests.

For exact per-LLM-call correlation, also set the HTTP `x-request-id` header to
the harness-generated `llm_call_id`. This ID is separate from `agent_context`:
`agent_context` groups requests into workflow/program lineage, while
`x-request-id` identifies one harness LLM call. Dynamo still records its own
`request_id` for the inference request.

```text
x-request-id: llm-call-42
```

```json
{
  "model": "my-model",
  "messages": [
    {
      "role": "user",
      "content": "Research the latest SGLang support for DeepSeek."
    }
  ],
  "nvext": {
    "agent_context": {
      "workflow_type_id": "deep_research",
      "workflow_id": "research-run-42",
      "program_id": "research-run-42:researcher",
      "parent_program_id": "research-run-42:planner"
    }
  }
}
```

| Field | Required | Description |
|-------|:--------:|-------------|
| `workflow_type_id` | Yes | Reusable profile or agent class label, such as `deep_research`, `coding_agent`, or `pi_mono`. |
| `workflow_id` | Yes | Workflow, run, or structured stage identifier. A structured workflow can include multiple programs under the same workflow. |
| `program_id` | Yes | One schedulable reason/tool trajectory, usually a single reason -> tool -> reason loop. |
| `parent_program_id` | No | Parent program that spawned this program. Use this for subagents. |

`agent_context` identifies the request. It is separate from
[`agent_hints`](../components/frontend/nvext.md#agent-hints), which describe
what Dynamo may do with the request.

Do not put workflow or program identity in `x-request-id`; use
`nvext.agent_context` for that. The deprecated `x-dynamo-request-id` header is
retained only for compatibility with older clients that supplied Dynamo request
IDs directly.

## Trace Sink

Set `DYN_AGENT_TRACE_JSONL` before starting Dynamo to write one JSON object per
normalized trace event.

```bash
export DYN_AGENT_TRACE_JSONL=/tmp/dynamo-agent-trace.jsonl
export DYN_AGENT_TRACE_CAPACITY=1024
export DYN_AGENT_TRACE_JSONL_BUFFER_BYTES=1048576
export DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS=1000
```

`DYN_AGENT_TRACE_CAPACITY` is optional and defaults to `1024`. It controls the
bounded in-process broadcast buffer used by the best-effort trace bus. Publishing
to the bus is non-blocking; if a consumer falls behind, records can be dropped
and Dynamo logs the lag.

The JSONL writer runs in a background task. It batches records with an async
buffer and flushes when either `DYN_AGENT_TRACE_JSONL_BUFFER_BYTES` is reached
or `DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS` elapses. Defaults are `1048576`
bytes and `1000` ms. Disk I/O is not on the LLM request hot path.

The sink is best-effort telemetry. It is intended for profiling and correlation,
not durable audit logging.

To ingest harness tool events through Dynamo's event plane, also enable the
tool-event subscriber:

```bash
export DYN_AGENT_TRACE_TOOL_EVENTS=1
export DYN_AGENT_TRACE_TOOL_EVENTS_TOPIC=agent-tool-events
export DYN_AGENT_TRACE_NAMESPACE=<dynamo-namespace>
```

`DYN_AGENT_TRACE_TOOL_EVENTS_TOPIC` defaults to `agent-tool-events`.
`DYN_AGENT_TRACE_NAMESPACE` is optional; when omitted, Dynamo uses the configured
model namespace or the local model endpoint namespace.

Harnesses publish normalized tool records to this topic using the Dynamo
event-plane transport. Dynamo subscribes to the topic and relays valid tool
records onto the same in-process trace bus as LLM request records.

## Trace Record

Dynamo emits a `request_end` event after the response stream completes or is
dropped.

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "request_end",
  "event_time_unix_ms": 1777312801000,
  "event_source": "dynamo",
  "agent_context": {
    "workflow_type_id": "deep_research",
    "workflow_id": "research-run-42",
    "program_id": "research-run-42:researcher",
    "parent_program_id": "research-run-42:planner"
  },
  "request": {
    "request_id": "cmpl-123",
    "x_request_id": "llm-call-42",
    "model": "my-model",
    "input_tokens": 4096,
    "output_tokens": 512,
    "cached_tokens": 3584,
    "request_received_ms": 1777312800000,
    "prefill_wait_time_ms": 12.3,
    "prefill_time_ms": 70.1,
    "ttft_ms": 82.4,
    "total_time_ms": 1000.1,
    "kv_hit_rate": 0.875,
    "kv_transfer_estimated_latency_ms": 3.2,
    "queue_depth": 3,
    "worker": {
      "prefill_worker_id": 1,
      "prefill_dp_rank": 0,
      "decode_worker_id": 1,
      "decode_dp_rank": 0
    }
  }
}
```

Optional fields are omitted when Dynamo did not observe them for a request.

Harnesses publish tool lifecycle records with the same schema to the configured
event-plane topic. Dynamo relays valid `tool_start`, `tool_end`, and
`tool_error` records onto the normalized trace bus:

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "event_source": "harness",
  "agent_context": {
    "workflow_type_id": "deep_research",
    "workflow_id": "research-run-42",
    "program_id": "research-run-42:researcher",
    "parent_program_id": "research-run-42:planner"
  },
  "tool": {
    "tool_call_id": "call-abc",
    "tool_class": "web_search",
    "status": "succeeded",
    "duration_ms": 420.5,
    "output_tokens": 96,
    "output_bytes": 2048
  }
}
```

## Correlating Events

Agent harnesses know tool boundaries. Dynamo knows LLM request timing and worker
placement. Use the shared `workflow_id`, `program_id`, and `parent_program_id`
fields to build the workflow tree, and use `x_request_id` to join a harness LLM
call to Dynamo request metrics when the harness sets `x-request-id`:

```text
harness:
  tool_start / tool_end / tool_error(agent_context)
  llm_call_id

Dynamo normalized trace bus:
  request_end(agent_context, request_id, x_request_id, tokens, timing, cache, worker)
  tool_start / tool_end / tool_error(agent_context, tool_call_id, duration, status)

joined profile:
  workflow -> programs -> LLM calls + tool calls + subagent lineage
```

This gives an offline profile of agent execution without changing request
serving behavior.

## Using The Trace

The first user journey is offline analysis. Run the agent harness and Dynamo
with Dynamo trace JSONL enabled, then read one normalized workflow timeline.

```text
Dynamo trace JSONL:
  request_end events with request_id, x_request_id, tokens, timing, cache, and worker fields
  tool_start / tool_end / tool_error events ingested from the event plane

Joined report:
  workflow timeline, program/subagent tree, LLM time, tool time, cache misses,
  prompt growth, queueing, and latency outliers
```

If the harness sets the `x-request-id` header, join LLM calls exactly:

```text
harness.llm_call_id == dynamo.request.x_request_id
```

Otherwise, join by `workflow_id`, `program_id`, and event ordering or nearest
timestamps. That is usually sufficient for sequential reason/tool loops, but it
can be ambiguous for retries, parallel LLM calls, cancellations, or subagents.

A useful first report should answer:

- Which programs and subagents ran under the workflow?
- How many LLM calls and tool calls did each program make?
- Where did wall time go: LLM prefill, decode, queueing, tool wait, or subagent wait?
- Which requests had the largest `input_tokens`, uncached tokens, TTFT, or total time?
- Did tool results or conversation history cause prompt growth between turns?
- Did repeated turns get cache hits, or did requests miss expected prefix cache?
- Which worker served each request?

For repeated runs, group reports by `workflow_type_id`. Those grouped traces are
the input to a workflow profile: expected LLM turns, output lengths, tool gaps,
cache behavior, and subagent fanout for that reusable workload class. Later
runtime features can use that profile to fill missing scheduling hints or choose
cache actions, but the current trace path is analysis-only.

## Current Scope

- `agent_context` is passive metadata in the current implementation.
- Dynamo emits request-end trace records when agent tracing is enabled.
- Dynamo can ingest harness tool lifecycle records from the event plane when
  `DYN_AGENT_TRACE_TOOL_EVENTS=1`.
- The trace sink is async local JSONL output behind a bounded bus. It does not
  publish durable audit records.
- Scheduling and cache-control decisions should continue to use existing
  request hints and router/backend mechanisms.

## See Also

- [NVIDIA Request Extensions](../components/frontend/nvext.md)
- [Agents](../features/agentic_workloads.md)
- [SGLang for Agentic Workloads](../backends/sglang/agents.md)
