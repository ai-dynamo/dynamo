---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Context
subtitle: Attach session and trajectory identity to agentic requests
---

Agent context is passive identity metadata for agentic requests. It lets a
harness label each LLM call with a top-level agent run and a specific
reasoning/tool trajectory. Dynamo records this metadata in agent traces, but it
does not change routing, scheduling, or cache behavior.

## Request Schema

Each harness LLM call should include `nvext.agent_context`:

```json
{
    "model": "my-model",
    "messages": [
        { "role": "user", "content": "Research Dynamo agent tracing." }
    ],
    "nvext": {
        "agent_context": {
            "session_type_id": "deep_research",
            "session_id": "research-run-42",
            "trajectory_id": "research-run-42:researcher",
            "parent_trajectory_id": "research-run-42:planner"
        }
    }
}
```

| Field                  | Required | Meaning                                                                     |
| ---------------------- | :------: | --------------------------------------------------------------------------- |
| `session_type_id`      |   Yes    | Reusable workload/profile class, such as `deep_research` or `coding_agent`. |
| `session_id`           |   Yes    | Top-level agent run/session identifier.                                     |
| `trajectory_id`        |   Yes    | One reasoning/tool trajectory within the agent run.                         |
| `parent_trajectory_id` |    No    | Parent trajectory for subagents.                                            |

A single `session_id` can contain multiple parent and child trajectories. The
field names align with the [Agent Trajectory Interchange Format][atif-rfc] so
harness trajectory files and Dynamo serving traces join without renaming; see
the collapsed section at the bottom of this page for details.

[atif-rfc]: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md

## OpenAI Client Integration

When using the OpenAI Python client, pass Dynamo's extension fields through
`extra_body` and set `x-request-id` through `extra_headers`:

```python
import uuid


def instrument_llm_request(kwargs, agent_context):
    body = dict(kwargs.get("extra_body") or {})
    nvext = dict(body.get("nvext") or {})
    nvext["agent_context"] = dict(agent_context)
    body["nvext"] = nvext

    headers = dict(kwargs.get("extra_headers") or {})
    headers.setdefault("x-request-id", str(uuid.uuid4()))

    out = dict(kwargs)
    out["extra_body"] = body
    out["extra_headers"] = headers
    return out
```

`x-request-id` is the harness's logical LLM-call ID. Dynamo copies it into
`request.x_request_id`; it is separate from Dynamo's internal request ID.

## Harness Integration Pattern

An existing harness does not need to import Dynamo packages or link against
Dynamo runtime APIs. Framework integrations should use this shape:

- Add a small helper module that stores the current `agent_context` in a context
  variable.
- Wrap each agent run with that context so LLM calls and tool records share the
  same `session_id` and `trajectory_id`.
- Call one helper before each OpenAI-compatible LLM request to merge
  `extra_body.nvext.agent_context` and set `x-request-id`.
- Propagate context through thread pools, subprocesses, and subagent launches
  when those paths can make LLM calls or emit tool records.
- Include `parent_trajectory_id` when launching a subagent from a known parent
  trajectory.

For trace sink setup, tool event relay, and record schema details, see
[Agent Tracing](agent-tracing.md).

<details>
<summary>ATIF alignment</summary>

The [Agent Trajectory Interchange Format (ATIF)][atif-rfc] is the JSON format
maintained as the [Harbor framework][harbor] data schema for complete agent
trajectories (user inputs, agent steps, tool calls, observations, subagents,
rewards). Dynamo does not emit ATIF; it emits `dynamo.agent.trace.v1`, a
serving-oriented trace covering request timing, tokens, cache, queue depth, and
worker placement. The two formats are complementary and join cleanly because
identifier names match:

| Dynamo field           | ATIF role                       | Meaning                                                         |
| ---------------------- | ------------------------------- | --------------------------------------------------------------- |
| `session_id`           | `session_id`                    | Agent run identity. Multiple trajectories share one session.    |
| `trajectory_id`        | `trajectory_id`                 | One parent or child trajectory within the run.                  |
| `parent_trajectory_id` | subagent relationship metadata  | Optional parent trajectory for subagents.                       |
| `session_type_id`      | producer-specific metadata      | Reusable workload/profile class.                                |

A harness ATIF file and Dynamo's trace stream can be joined offline on
`session_id` + `trajectory_id` without schema changes. Full ATIF reconstruction
still requires harness trajectory data; Dynamo trace records intentionally omit
prompt and response content.

[harbor]: https://github.com/harbor-framework/harbor

</details>
