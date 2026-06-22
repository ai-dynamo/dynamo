---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

NVIDIA Dynamo optimizes agent workloads through a lightweight set of headers and request extensions that affect the behavior of the router, the inference engine and the the KV cache manager. The harness remains responsible for agent semantics. Dynamo receives lightweight metadata and uses it for observability, replay, routing hints, priority, and cache-aware serving.

The common identity concept is `trajectory_id`: one stable ID for one agent reasoning/tool chain. Supported coding agents can rely on the HTTP headers they already emit, and custom clients can send Dynamo trajectory headers. See [Trajectory IDs](trajectory-ids.md#trajectory-id-inputs) for the exact contract.

## Core Concepts

| Concept | Purpose |
|---------|---------|
| [Dynamo for Agents](introduction.md) | Conceptual model for trajectories, requests, tool events, and the split between passive identity and active serving policy. |
| [Trajectory IDs](trajectory-ids.md) | Stable agent trajectory identity from first-class coding-agent headers or Dynamo trajectory headers. |
| [Agent Tracing](agent-tracing.md) | Request trace output, inferred tool-call metadata, optional harness tool spans, Perfetto conversion, and request replay. |
| [Agent Hints](agent-hints.md) | Optional per-request hints such as priority, expected output length, and speculative prefill. |
| [Priority Scheduling](priority-scheduling.md) | Request priority semantics across the router queue, backend engines, and cache policy. |
| [Use Pi-Mono with Dynamo](pi-mono.md) | End-to-end quickstart that drives the Pi coding agent through Dynamo with trajectory identity and tool tracing turned on. |
| [ThunderAgent Program Scheduler](thunderagent-router.md) | Experimental scheduler keyed by trajectory identity with tool-boundary pause/resume on top of KV-aware routing: the 5s scheduler tick, the utilization-driven control loop and its knobs, and scheduler observability. |

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