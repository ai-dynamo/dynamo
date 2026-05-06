---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: ATIF Alignment
subtitle: Join Dynamo serving telemetry with agent trajectory traces
---

The [Agent Trajectory Interchange Format (ATIF)][atif-rfc] is a JSON format for
complete agent trajectories: user inputs, agent steps, tool calls, observations,
subagents, rewards, and evaluation metadata. ATIF is maintained as the
[Harbor framework][harbor] data format and is the canonical source of truth for
the schema referenced below.

Dynamo does not emit ATIF in the current trace path. Dynamo emits
`dynamo.agent.trace.v1`, a serving-oriented trace that records request timing,
token counts, cache metrics, queue depth, worker placement, and harness tool
events. The two formats are complementary.

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
[harbor]: https://github.com/harbor-framework/harbor

## Identifier Alignment

Dynamo uses ATIF-aligned identifier names in `nvext.agent_context`:

| Dynamo field | ATIF role | Meaning |
|--------------|-----------|---------|
| `session_id` | `session_id` | Agent run identity. Multiple trajectories can share one session. |
| `trajectory_id` | `trajectory_id` | One parent or child trajectory within the run. |
| `parent_trajectory_id` | subagent relationship metadata | Optional parent trajectory for subagents. |
| `session_type_id` | producer-specific metadata | Reusable workload or profile class. |

This lets an ATIF file from a harness or eval framework join with Dynamo's
serving trace without renaming fields.

```text
ATIF trajectory
  session_id = research-run-42
  trajectory_id = research-run-42:searcher
        |
        | join on session_id + trajectory_id
        v
Dynamo trace records
  request timing, tokens, cache hit rate, queue depth, worker placement
```

## Ownership Boundary

Harnesses and eval frameworks should own semantic trajectory capture. They see
the prompts, observations, actions, tool arguments, tool outputs, rewards, and
subagent structure needed for debugging, SFT, RL, and evaluation.

Dynamo should own inference-serving telemetry. It sees the queueing, prefill,
decode, token, cache, routing, and worker-placement data needed to explain how
the inference engine served each trajectory.

Keeping these boundaries separate avoids forcing Dynamo's low-level serving
trace to become a payload-heavy trajectory log. Full-fidelity ATIF export should
come from the harness or from an offline join of harness/audit data with Dynamo
trace records.

## Current Dynamo Scope

- Dynamo aligns identifier names with ATIF.
- Dynamo does not write native ATIF files in this trace path.
- Dynamo trace records intentionally omit prompt and response content.
- Full ATIF reconstruction requires harness trajectory data, audit payload data,
  or both.
- Future exporters can store Dynamo-specific metrics in [ATIF `extra` fields][atif-rfc]
  or produce a separate sidecar joined by `session_id` and `trajectory_id`.
