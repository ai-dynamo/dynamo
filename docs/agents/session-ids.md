---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Session IDs
subtitle: Identify agent sessions from supported coding agents and custom clients
---

A session ID is the stable identifier Dynamo uses for one agent reasoning/tool chain. A root agent, planner, researcher subagent, or OpenCode subtask can each have its own session. Every LLM request in that chain should carry the same `session_id`; child sessions can also carry a `parent_session_id` so traces and replay tools can rebuild the tree. Some academic papers also call this a `program_id`.

Session identity is passive metadata unless session affinity is explicitly enabled.
Sending `X-Dynamo-Session-ID` alone does not change request placement. Tracing records
the identity when `DYN_REQUEST_TRACE` is enabled. When
`--router-session-affinity-ttl-secs` is configured, the router uses the ID for an
immutable endpoint- and phase-scoped worker binding.

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
| Claude Code | `x-claude-code-session-id`; `x-claude-code-agent-id` for child agents | `x-claude-code-parent-agent-id`; falls back to `x-claude-code-session-id` | Root turns use the session header as `session_id`; child-agent turns use the agent header as `session_id`. Nested children use the parent-agent header as `parent_session_id`; top-level children use the root session header. |
| Codex | `session-id` | None | `session-id` becomes the `session_id`. |
| OpenCode | `x-session-id` | `x-parent-session-id` | `x-session-id` becomes the `session_id`; `x-parent-session-id` becomes `parent_session_id` when present. |

`X-Dynamo-Session-Final` applies with either canonical or agent-native session
identity. With session affinity enabled, a final request routes normally and then
terminally closes its binding. Close invalidation across replicas is eventual. Do not
send more requests with that session ID after close.

etcd and FileStore on a shared filesystem coordinate bindings across frontend
processes. MemoryStore and Kubernetes discovery retain process-local affinity only.
The affinity TTL controls local cache cleanup, not the distributed claim lifetime.
Claims follow the creating frontend's etcd lease or FileStore ownership. If a claim
expires or its bound worker disappears, create a new session with a new ID instead of
reusing or rebinding the old ID. See [Router session affinity](../components/router/router-configuration.md#session-affinity)
for the full contract.

### Custom Agent Harnesses

For a custom HTTP client that only needs a session ID, send the generic header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-session-id: research-run-42:researcher' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
```
