---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Context and Tracing
subtitle: Attach workflow identity to agentic requests
---

Dynamo currently supports passive agent request tracing. An agent harness can
attach identity metadata to each LLM request, and Dynamo can write normalized
request-end records to a local JSONL sink. Tool lifecycle events can also be
relayed into the same trace bus through a Dynamo-owned ZMQ relay.

This is observability only. It does not change routing, scheduling, or cache
behavior.

## Request Metadata

Set `nvext.agent_context` on chat completion requests:

```json
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Research Dynamo agent tracing."}],
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

For exact per-call correlation, set the HTTP `x-request-id` header to the
harness LLM call ID:

```text
x-request-id: llm-call-42
```

`x-request-id` is not Dynamo's internal inference request ID. It is copied into
the trace record as `request.x_request_id`.

| Field | Required | Meaning |
|-------|:--------:|---------|
| `workflow_type_id` | Yes | Reusable workload/profile class, such as `deep_research` or `coding_agent`. |
| `workflow_id` | Yes | Top-level run identifier. |
| `program_id` | Yes | One schedulable reasoning/tool trajectory. |
| `parent_program_id` | No | Parent program for subagents. |

## Enabling JSONL Output

Set `DYN_AGENT_TRACE_JSONL` before starting Dynamo:

```bash
export DYN_AGENT_TRACE_JSONL=/tmp/dynamo-agent-trace.jsonl
export DYN_AGENT_TRACE_CAPACITY=1024
```

Dynamo writes one JSON object per normalized trace event. The sink is
best-effort local telemetry for debugging and offline profiling. It is not a
durable audit log.

## Request-End Record

Dynamo emits `request_end` after the response stream completes or is dropped.
Nullable fields are omitted when the serving path did not record them.

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
    "request_id": "dynamo-request-id",
    "x_request_id": "llm-call-42",
    "model": "my-model",
    "input_tokens": 4096,
    "output_tokens": 512,
    "cached_tokens": 3584,
    "request_received_ms": 1777312800000,
    "ttft_ms": 82.4,
    "total_time_ms": 1000.1,
    "kv_hit_rate": 0.875,
    "queue_depth": 3
  }
}
```

## Tool Event Relay

For harness tool events, Dynamo owns the event-plane publishing path. Start
Dynamo with a local ZMQ source endpoint:

```bash
export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390
export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC=
```

The ZMQ wire format is:

```text
[topic, seq_be_u64, msgpack(AgentTraceRecord)]
```

Only harness-originated `tool_start`, `tool_end`, and `tool_error` records are
accepted. Dynamo republishes valid records onto the runtime event plane and then
feeds them into the same in-process trace bus as request-end records.

Tool events use the same envelope:

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777312801500,
  "event_source": "harness",
  "agent_context": {
    "workflow_type_id": "deep_research",
    "workflow_id": "research-run-42",
    "program_id": "research-run-42:researcher"
  },
  "tool": {
    "tool_call_id": "call-abc",
    "tool_class": "web_search",
    "status": "succeeded",
    "duration_ms": 420.5,
    "output_bytes": 2048
  }
}
```

## Current Scope

- `agent_context` is passive metadata.
- Dynamo emits request-end trace records when agent tracing is enabled.
- Tool events enter through a Dynamo-owned ZMQ relay, not through a generic
  Python event-plane publisher.
- JSONL is a local debug/profiling sink.
- Future scheduler/profiler consumers should read the normalized trace bus.
