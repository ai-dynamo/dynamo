---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

NVIDIA Dynamo optimizes agent workloads with lightweight headers and request extensions for the router, inference engine, and KV cache manager. The harness remains responsible for agent semantics, while Dynamo uses request metadata for observability, replay, routing, priority, and cache-aware serving.

| Layer | Signal | Optimization |
|-------|--------|--------------|
| Frontend API | Session and trajectory headers plus `nvext` request extensions | Normalize affinity, agent identity, and serving intent across APIs. |
| Router | Session affinity, priority, expected output length, and cache-overlap signals | Pin sessions, place other requests for KV reuse, and order queued work. |
| KV cache management | Priority and trajectory metadata forwarded to the backend runtime | Influence engine scheduling, cache eviction, and trajectory-aware radix tagging where the backend supports it. |

Use `trajectory_id` as one stable identifier for an agent reasoning/tool chain.
Send `x-dynamo-trajectory-id` explicitly, or use Claude Code's agent ID. Native
harness session headers remain separate router-affinity identifiers. See
[Trajectory IDs](trajectory-ids.md#trajectory-id-inputs) for the exact contract.

## Documentation

| Concept | Purpose |
|---------|---------|
| [Agent Harnesses](agent-harnesses.md) | Quickstart for running popular agent harnesses through Dynamo. |
| [Trajectory IDs](trajectory-ids.md) | Stable agent identity for scheduling, tracing, and more |
| [Agent Tracing](agent-tracing.md) | Request traces, inferred tool calls, optional harness tool spans, and Perfetto conversion. |
| [Agent Simulation](agent-replay.md) | Convert agent traces into replay and simulation inputs. |
| [Agent Hints](agent-hints.md) | Per-request hints such as priority, expected output length, and speculative prefill. |
| [Priority Scheduling](../components/router/priority-scheduling.md) | Priority behavior across the router queue, backend engines, and cache policy. |
| [ThunderAgent Program Scheduler](thunderagent-router.md) | Experimental tool-boundary pause/resume scheduler on top of KV-aware routing. |

## Request Surface

Agent trajectory identity is header-only. Agent-facing body metadata under `nvext` is for hints and controls.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-trajectory-id: research-run-42:researcher' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "..."}],
    "nvext": {
      "agent_hints": {
        "priority": 5,
        "osl": 1024
      }
    }
  }'
```

Use trajectory IDs when you want traceability across LLM calls, tool calls, and external trajectory files. Use `agent_hints` when you want to influence serving behavior at the router and engine layer.
