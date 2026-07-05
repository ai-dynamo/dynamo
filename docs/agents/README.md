---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

NVIDIA Dynamo optimizes agent workloads with lightweight headers and request extensions for the router, inference engine, and KV cache manager. The harness remains responsible for agent semantics, while Dynamo uses request metadata for observability, replay, routing, priority, and cache-aware serving.

| Layer | Signal | Optimization |
|-------|--------|--------------|
| Frontend API | Session headers and `nvext` request extensions | Normalize agent identity and serving intent across APIs. |
| Router | Priority, expected output length, and cache-overlap signals | Place requests for KV reuse and order queued work. |
| KV cache management | Priority and session metadata forwarded to the backend runtime | Influence engine scheduling, cache eviction, and subagent KV isolation where the backend supports it. |

The common identity concept is `session_id`: one stable ID for one agent reasoning/tool chain. Dynamo maps supported coding-agent headers to `session_id`, and custom harnesses can send `X-Dynamo-Session-ID` directly. The ID is passive metadata: it does not enable sticky sessions or session-aware routing. A routing policy must opt in to use it. See [Session IDs](session-ids.md#session-id-inputs) for the exact contract.

## Documentation

| Concept | Purpose |
|---------|---------|
| [Deploy a Mixture of Models](deploy-mixture-of-models.md) | Route agent requests between heterogeneous model deployments through one endpoint. |
| [Agent Harnesses](agent-harnesses.md) | Quickstart for running popular agent harnesses through Dynamo. |
| [Session IDs](session-ids.md) | Stable agent identity for tracing and opt-in consumers. |
| [Agent Tracing](agent-tracing.md) | Request traces, inferred tool calls, optional harness tool spans, and Perfetto conversion. |
| [Agent Simulation](agent-replay.md) | Convert agent traces into replay and simulation inputs. |
| [Agent Hints](agent-hints.md) | Per-request hints such as priority, expected output length, and speculative prefill. |
| [Priority Scheduling](../components/router/priority-scheduling.md) | Priority behavior across the router queue, backend engines, and cache policy. |
| [ThunderAgent Program Scheduler](thunderagent-router.md) | Experimental tool-boundary pause/resume scheduler on top of KV-aware routing. |

## Request Surface

Agent session identity is header-only. Agent-facing body metadata under `nvext` is for hints and controls.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
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

Use session IDs when you want traceability across LLM calls, tool calls, and external trajectory files. Use `agent_hints` when you want to influence serving behavior at the router and engine layer. Configure session-aware routing separately when a routing policy supports it.
