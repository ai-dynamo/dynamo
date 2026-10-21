---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Serve coding agents and multi-step agent sessions faster
---

Agent traffic looks different from chat. Each turn repeats a growing prefix, tool calls leave the GPU idle between turns, and some requests (the one a user is waiting on) matter more than others (a background subagent). Dynamo does not run your agent loop. Your harness still owns prompts, tools, subagents, and reasoning state. Dynamo reads a little metadata on each request and uses it to route and schedule better, keep reusable KV cache, and record what happened.

## What You Can Do

| I want to… | Use | Guide |
|---|---|---|
| Point a coding agent at Dynamo | A local endpoint and a harness config | [Run an agent on Dynamo](run-an-agent-on-dynamo.md) |
| Tell Dynamo which requests belong to which agent | Session IDs | [Identify agent sessions](session-ids.mdx) |
| Keep a session's turns on the worker that holds its KV cache | Session affinity | [Maintain session affinity](session-affinity.md) |
| Let interactive requests go ahead of background work | Priority hints | [Prioritize requests](priority-scheduling.mdx) |
| See where time goes between LLM calls and tools | Request tracing | [Agent tracing](agent-tracing.md) |
| Compare serving configurations without rerunning the agent | Trace replay | [Agent simulation](agent-simulation.mdx) |
| Free KV cache held by agents that are waiting on tools | ThunderAgent | [ThunderAgent Program Scheduler](thunderagent-program-scheduler.md) |

## Try It

Send a request that carries a session ID and a priority hint:

```bash {4,9-11}
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "..."}],
    "nvext": {
      "agent_hints": { "priority": 5, "osl": 1024 }
    }
  }'
```

The response is a normal chat completion. The session ID labels the request for tracing. The hints change queueing, scheduling, and cache eviction only under load, and only for the priority layers you turn on. With the launcher defaults they are metadata. See [Prioritize requests](priority-scheduling.mdx) for the switches. Neither one pins the session to a worker; for that, see [Maintain session affinity](session-affinity.md).

## How Dynamo Uses the Metadata

![The agent harness sends each LLM request with a session ID and optional hints. The Dynamo frontend normalizes the metadata, the router uses it for placement, session binding, and queue order, and the backend engine uses priority for scheduling and eviction. The frontend also records a request trace.](../../../assets/img/agents-metadata-flow.svg)

| Layer | Signal | What Dynamo does with it |
|---|---|---|
| Frontend API | Session headers and `nvext` request extensions | Normalizes agent identity and serving intent across the Chat Completions, Responses, and Messages APIs. |
| Router | Session ID, priority, expected output length, cache overlap | Places requests for KV reuse, binds sessions to workers when affinity is on, and orders queued work. |
| Backend engine | Priority forwarded by the router | Influences engine scheduling and cache eviction where the backend supports it. |

## Where to Start

1. Follow [Run an agent on Dynamo](run-an-agent-on-dynamo.md) to get an agent talking to a local deployment.
2. Add [session IDs](session-ids.mdx) so every request is attributable.
3. Turn on [session affinity](session-affinity.md) if your agents run many turns on the same context.
4. Add [priority](priority-scheduling.mdx) once interactive and background requests compete for the same workers.
5. Use [tracing](agent-tracing.md) to measure each change before you keep it.

For SGLang engine flags and subagent KV isolation, see [Agents on SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/agents-on-sglang.md).
