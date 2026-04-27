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
The result is a JSONL trace that can be correlated with harness-side tool-call
events.

## Request Contract

Set `nvext.agent_context` on chat completion requests.

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

## Trace Sink

Set `DYN_AGENT_TRACE_JSONL` before starting Dynamo to write one JSON object per
completed agent request.

```bash
export DYN_AGENT_TRACE_JSONL=/tmp/dynamo-agent-trace.jsonl
export DYN_AGENT_TRACE_CAPACITY=1024
```

`DYN_AGENT_TRACE_CAPACITY` is optional and defaults to `1024`. It controls the
in-process broadcast buffer used by the best-effort trace sink.

The sink is best-effort telemetry. It is intended for profiling and correlation,
not durable audit logging.

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

## Correlating With Tool Events

Agent harnesses usually know when tools start and finish. Dynamo knows when LLM
requests start, finish, and which workers served them. Use the shared
`workflow_id`, `program_id`, and `parent_program_id` fields to join both views:

```text
harness tool events:
  tool_start / tool_end(program_id, parent_program_id)

Dynamo request events:
  request_end(agent_context, tokens, timing, cache, worker)

joined profile:
  workflow -> programs -> LLM calls + tool calls + subagent lineage
```

This gives an offline profile of agent execution without changing request
serving behavior.

## Current Scope

- `agent_context` is passive metadata in the current implementation.
- Dynamo emits request-end trace records only when `DYN_AGENT_TRACE_JSONL` is
  set.
- The trace sink is local JSONL output. It does not publish durable audit
  records.
- Scheduling and cache-control decisions should continue to use existing
  request hints and router/backend mechanisms.

## See Also

- [NVIDIA Request Extensions](../components/frontend/nvext.md)
- [Agents](../features/agentic_workloads.md)
- [SGLang for Agentic Workloads](../backends/sglang/agents.md)
