---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

Dynamo provides a small set of request extensions and trace utilities for
serving agentic workloads. The harness remains responsible for the semantic
agent trajectory. Dynamo receives lightweight metadata and uses it for serving
telemetry, routing hints, and backend-specific cache behavior.

## Core Concepts

| Concept | Purpose |
|---------|---------|
| [Agent Context](agent-context.md) | Passive `session_id` and `trajectory_id` metadata attached to OpenAI-compatible requests. |
| [Agent Tracing](agent-tracing.md) | Dynamo-owned request timing, token, cache, worker-placement, and harness tool-event traces. |
| [Agent Hints](agent-hints.md) | Optional per-request hints such as priority, expected output length, and speculative prefill. |
| [Tool Calling](tool-calling.md) | Supported tool-call parsers and parser names. |
| [Reasoning](reasoning.md) | Supported reasoning parsers for chain-of-thought models. |
| [Chat Processors](chat-processor-options.md) | Dynamo, vLLM, and SGLang preprocessing options. |

## Backend-Specific Guides

Agent features are exposed through common request metadata, but backend support
varies by runtime.

| Backend Guide | Contents |
|---------------|----------|
| [SGLang for Agentic Workloads](../backends/sglang/agents.md) | Priority scheduling, priority-based radix eviction, speculative prefill, and streaming session control for subagent KV isolation. |

## Request Surface

Agent-facing request metadata lives under `nvext` on OpenAI-compatible request
bodies:

```json
{
    "nvext": {
        "agent_context": {
            "session_type_id": "deep_research",
            "session_id": "research-run-42",
            "trajectory_id": "research-run-42:researcher"
        },
        "agent_hints": {
            "priority": 5,
            "osl": 1024
        }
    }
}
```

Use `agent_context` when you want traceability across LLM calls, tool calls, and
external trajectory files. Use `agent_hints` only when the harness has
serving-relevant intent that Dynamo can act on.
