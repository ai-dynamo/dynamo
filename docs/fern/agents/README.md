---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agents
subtitle: Agent-aware serving features in Dynamo
---

NVIDIA Dynamo adds agent-aware serving features without taking ownership of the agent loop: your harness still manages prompts, tools, subagents, and reasoning state, while Dynamo uses metadata attached to each LLM request to correlate work, improve routing and scheduling, manage KV cache behavior, and produce traces for replay and analysis.

## Send Your First Agent Request

```bash title="Chat completions with session ID and agent hints" {4,9-12}
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

<Info>
Session IDs identify work for tracing and opt-in consumers. Agent hints influence serving behavior. Neither enables sticky placement unless a separate routing policy is configured.
</Info>

## Choose Your Metadata

| Metadata | Where | Role | Use when |
|----------|-------|------|----------|
| Session ID | Request header | Passive identity | You need traceability across LLM calls, tool spans, and replay |
| Agent hints | `nvext.agent_hints` in the request body | Active serving intent | You want to influence queue order, scheduling, or cache behavior |

See [Session IDs](session-ids.mdx#session-id-inputs) and [Agent Hints](agent-hints.md) for the full contract.

## Implementation Checklist

<Steps toc={true}>

<Step title="Configure a packaged harness when available">

Point Codex, Pi, Claude Code, or another supported CLI at Dynamo. Supported harnesses emit native session headers, so you do not need to add `X-Dynamo-Session-ID` yourself. See [Agent Harnesses](agent-harnesses.md).

</Step>

<Step title="Identify agent sessions from a custom client">

If you are building a custom client instead of using a packaged harness, send `X-Dynamo-Session-ID` on every request in a reasoning chain. See [Session IDs](session-ids.mdx).

</Step>

<Step title="Add agent hints when serving intent matters">

Set `nvext.agent_hints` only when you want router or engine behavior to change for that request. See [Agent Hints](agent-hints.md).

</Step>

<Step title="Enable tracing if you need measurements">

Set `DYN_REQUEST_TRACE=1` on the frontend to capture timing, tool calls, and session identity. See [Agent Tracing](agent-tracing.md).

</Step>

</Steps>

## Where Signals Are Used

| Layer | Signal | Optimization |
|-------|--------|--------------|
| Frontend API | Session headers and `nvext` request extensions | Normalize agent identity and serving intent across APIs. |
| Router | Priority, expected output length, and cache-overlap signals | Place requests for KV reuse and order queued work. |
| KV cache management | Priority and session metadata forwarded to the backend runtime | Influence engine scheduling, cache eviction, and subagent KV isolation where the backend supports it. |

## Next Steps

<CardGroup cols={3}>
  <Card title="Session IDs" icon="regular fingerprint" href="session-ids.mdx">
    Stable agent identity for tracing and opt-in consumers.
  </Card>
  <Card title="Agent Hints" icon="regular sliders" href="agent-hints.md">
    Per-request priority, expected output length, and speculative prefill.
  </Card>
  <Card title="Agent Harnesses" icon="regular terminal" href="agent-harnesses.md">
    Connect Codex, Pi, Claude Code, and other supported coding agents.
  </Card>
  <Card title="Agent Tracing" icon="regular chart-line" href="agent-tracing.md">
    Request traces, inferred tool calls, and Perfetto conversion.
  </Card>
  <Card title="Agent Simulation" icon="regular play" href="agent-replay.md">
    Convert agent traces into replay and simulation inputs.
  </Card>
  <Card title="ThunderAgent Program Scheduler" icon="regular clock" href="thunderagent-router.md">
    Experimental tool-boundary pause/resume scheduler on top of KV-aware routing.
  </Card>
</CardGroup>
