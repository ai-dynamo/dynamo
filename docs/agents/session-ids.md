---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Session IDs
subtitle: Identify agent sessions from supported coding agents and custom clients
---

A session ID is the stable identifier Dynamo uses for one agent reasoning/tool chain. A root agent, planner, researcher subagent, or OpenCode subtask can each have its own session. Every LLM request in that chain should carry the same `session_id`; child sessions can also carry a `parent_session_id` so traces and replay tools can rebuild the tree. Some academic papers also call this a `program_id`.

Session identity is passive metadata. Sending `X-Dynamo-Session-ID` does not enable sticky sessions or change request placement. Tracing records the identity when `DYN_REQUEST_TRACE` is enabled, and a session-aware routing policy can consume it only when that policy is configured separately.

## Session ID Inputs

Custom clients should send the canonical Dynamo headers. When `X-Dynamo-Session-ID` is present, Dynamo uses it and `X-Dynamo-Parent-Session-ID` instead of any agent-native identity values.

| Header | Normalized `agent_context` field | Required | Meaning |
|--------|----------------------------------|:--------:|---------|
| `X-Dynamo-Session-ID` | `session_id` | Yes | One reasoning/tool chain inside the run. |
| `X-Dynamo-Parent-Session-ID` | `parent_session_id` | No | Parent session when using subagents. |
| `X-Dynamo-Session-Final` | `session_final` | No | `true` marks the session's last request for lifecycle-aware consumers. |

### Native Agent Headers

Dynamo also recognizes the current stable identity headers emitted by the following coding agents. The [frontend API surface compliance test](https://github.com/ai-dynamo/dynamo/blob/main/tests/frontend/test_frontend_api_surface_compliance.py) catches header changes as coding agents evolve.

| Source | Session input | Parent input | Dynamo behavior |
|--------|------------------|--------------|-----------------|
| Claude Code | `x-claude-code-session-id`; `x-claude-code-agent-id` for child agents | Inferred from `x-claude-code-session-id` when `x-claude-code-agent-id` differs | Root turns use the session header as `session_id`; child-agent turns use the agent header as `session_id` and the session header as `parent_session_id`. |
| Codex | `session-id` | None | `session-id` becomes the `session_id`. |
| OpenCode | `x-session-id` | `x-parent-session-id` | `x-session-id` becomes the `session_id`; `x-parent-session-id` becomes `parent_session_id` when present. |

`X-Dynamo-Session-Final` applies with either canonical or agent-native session identity.

### Custom Agent Harnesses

For a custom HTTP client that only needs a session ID, send the generic header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
```
