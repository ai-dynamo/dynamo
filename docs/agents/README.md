---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

NVIDIA Dynamo optimizes agent workloads with lightweight headers and request extensions for the router, inference engine, and KV cache manager. These headers and extensions are non-invasive and extensible: the harness remains responsible for agent semantics, while Dynamo uses request metadata for observability, replay, routing, priority, and cache-aware serving.

| Layer | Signal | Optimization |
|-------|--------|--------------|
| Frontend API | Trajectory headers and `nvext` request extensions | Normalize agent identity and serving intent across APIs. |
| Router | Trajectory identity, priority, expected output length, and cache-overlap signals | Place requests for KV reuse, order queued work, and support agent-aware routing strategies. |
| KV cache management | Priority and session metadata forwarded to the backend runtime | Influence engine scheduling, cache eviction, and subagent KV isolation where the backend supports it. |

The common identity concept is `trajectory_id`: one stable ID for one agent reasoning/tool chain. For popular coding agents, Dynamo internally tranlates their header IDs to an internal `trajectory_id`. For custom harnesses, all you have to add is a `x-dynamo-trajectory-id` to take advantage of the entire stack. See [Trajectory IDs](trajectory-ids.md#trajectory-id-inputs) for more information.

## Core Concepts

| Concept | Purpose |
|---------|---------|
| [Dynamo for Agents](introduction.md) | Conceptual model for trajectories, requests, tool events, and serving policy. |
| [Trajectory IDs](trajectory-ids.md) | Stable agent identity from coding-agent headers or Dynamo trajectory headers. |
| [Agent Tracing](agent-tracing.md) | Request traces, inferred tool calls, optional harness tool spans, Perfetto conversion, and replay. |
| [Agent Hints](agent-hints.md) | Per-request hints such as priority, expected output length, and speculative prefill. |
| [Priority Scheduling](priority-scheduling.md) | Priority behavior across the router queue, backend engines, and cache policy. |
| [Use Pi-Mono with Dynamo](pi-mono.md) | Quickstart for running the Pi coding agent through Dynamo with trajectory identity and tool tracing. |
| [ThunderAgent Program Scheduler](thunderagent-router.md) | Experimental tool-boundary pause/resume scheduler on top of KV-aware routing. |

## Backend-Specific Guides

Agent features are exposed through common request metadata, but backend support varies by runtime.

| Backend Guide | Contents |
|---------------|----------|
| [SGLang for Agentic Workloads](../backends/sglang/agents.md) | Priority scheduling, priority-based radix eviction, speculative prefill, and streaming session control for subagent KV isolation. |

## Request Surface

Agent trajectory identity is header-only. Agent-facing body metadata under `nvext` is for hints and controls.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-trajectory-id: research-run-42:researcher' \
  -H 'x-dynamo-parent-trajectory-id: research-run-42:planner' \
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
