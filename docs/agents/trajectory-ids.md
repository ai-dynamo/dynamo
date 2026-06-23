---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Trajectory IDs
subtitle: Identify agent trajectories from supported coding agents and custom clients
---

A trajectory ID is the stable identifier Dynamo uses for one agent reasoning/tool chain. A root agent, planner, researcher subagent, or OpenCode subtask can each have its own trajectory. Every LLM request in that chain should carry the same `trajectory_id`; child trajectories can also carry a `parent_trajectory_id` so traces and replay tools can rebuild the tree. Some academic papers also call this a `program_id`.

## Trajectory ID inputs

### First-class supported agents

Dynamo recognizes explicit trajectory headers from Dynamo clients and Claude Code.
Native harness session headers identify router affinity instead of trajectories.

| Source | Trajectory input | Parent input | Dynamo behavior |
|--------|------------------|--------------|-----------------|
| Claude Code | `x-claude-code-agent-id` | None | Becomes `trajectory_id` when `x-dynamo-trajectory-id` is absent. `x-claude-code-session-id` remains affinity-only. |
| Codex | None | None | `session-id` remains affinity-only. Send Dynamo trajectory headers to add agent identity. |
| OpenCode | None | None | `x-session-id` remains affinity-only. `x-parent-session-id` does not create a parent trajectory. |
| Generic Dynamo client | `x-dynamo-trajectory-id` | `x-dynamo-parent-trajectory-id` | The canonical trajectory header takes precedence over Claude agent identity. Parent and final fields apply only when a trajectory exists. |

### Custom agent harnesses

For a custom HTTP client that only needs a trajectory ID, send the generic header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dummy' \
  -H 'x-dynamo-trajectory-id: research-run-42:researcher' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"..."}]}'
```

| Header | Required | Meaning |
|--------|:--------:|---------|
| `x-dynamo-trajectory-id` | Yes | One reasoning/tool chain inside the run. |
| `x-dynamo-parent-trajectory-id` | No | Parent trajectory when using subagents. |
| `x-dynamo-trajectory-final` | No | `true` marks the trajectory's last request for trace and KV-hint consumers. |

Trajectory headers never create router affinity. To keep requests on one worker,
send `X-Dynamo-Session-ID` separately. See
[Session Affinity](../components/router/router-configuration.md#session-affinity).
