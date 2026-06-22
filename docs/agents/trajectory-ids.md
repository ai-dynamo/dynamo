---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Trajectory IDs
subtitle: Identify agent trajectories from supported coding agents and custom clients
---

A trajectory ID is the stable identifier Dynamo uses for one agent reasoning/tool chain. A root agent, planner, researcher subagent, or OpenCode subtask can each have its own trajectory. Every LLM request in that chain should carry the same `trajectory_id`; child trajectories can also carry a `parent_trajectory_id` so traces and replay tools can rebuild the tree. Some academic papers also call this a `program_id`.

## Trajectory ID inputs

### First-class supported agents

Dynamo recognizes the current stable identity headers emitted by the following coding agents. The [frontend API surface compliance test](https://github.com/ai-dynamo/dynamo/blob/main/tests/frontend/test_frontend_api_surface_compliance.py) catches header changes as coding agents evolve.

| Source | Trajectory input | Parent input | Dynamo behavior |
|--------|------------------|--------------|-----------------|
| Claude Code | `x-claude-code-session-id`; `x-claude-code-agent-id` for child agents | Inferred from `x-claude-code-session-id` when `x-claude-code-agent-id` differs | Root turns use the session header as `trajectory_id`; child-agent turns use the agent header as `trajectory_id` and the session header as `parent_trajectory_id`. |
| Codex | `session-id` | None | `session-id` becomes the `trajectory_id`. |
| OpenCode | `x-session-id` | `x-parent-session-id` | `x-session-id` becomes the `trajectory_id`; `x-parent-session-id` becomes `parent_trajectory_id` when present. |
| Generic Dynamo client | `x-dynamo-trajectory-id` | `x-dynamo-parent-trajectory-id` | The header value becomes `trajectory_id`; the parent header becomes `parent_trajectory_id` when present. |

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
| `x-dynamo-trajectory-final` | No | `true` marks the trajectory's last request for lifecycle-aware consumers. |
