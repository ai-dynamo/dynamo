---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Harnesses
subtitle: Point coding-agent CLIs at a Dynamo deployment
---

Dynamo exposes `v1/chat/completions`, `v1/responses`, and `v1/messages` so **any agent** that uses these APIs can talk to a Dynamo endpoint. This guide specifically focuses on popular agent harnesses that send stable trajectory ID that Dynamo use for optimized routing and scheduling. 

## Local Setup

To locally test this out, we have a small script tha runs an SGLang-backed GLM 4.7 Flash endpoint. This script starts a TP2 instance on port 8000 and enables request tracing for replay and visualization. By default traces are saved in `/tmp/dynamo-request-trace-$(date +%Y%m%d-%H%M%S)`

To start it, run: 

```bash
bash examples/backends/sglang/launch/agg_agent.sh
```

## Codex

Codex uses the Responses API. Add a local provider in `~/.codex/config.toml`:

```toml
[model_providers.dynamo]
name = "dynamo"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "DYNAMO_API_KEY"
```

Then set one API-key env var and run Codex against the Dynamo model name:

```bash
codex -m zai-org/GLM-4.7-Flash \
  -c model_provider=dynamo \
  exec "Say ok"
```

Codex sends a `session-id` header that is internally mapped to our `trajectory_id`

## Claude Code

Claude Code should use Dynamo's Anthropic-compatible Messages API. The local launcher above starts `dynamo.frontend` with `--enable-anthropic-api`; for other deployments, pass that flag when starting the frontend. Then set:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=dummy

claude --model zai-org/GLM-4.7-Flash -p "Say ok"
```

For longer Claude Code sessions, also set `DYN_STRIP_ANTHROPIC_PREAMBLE=1` on the Dynamo frontend to remove Claude Code's billing preamble from the prompt before routing. Dynamo maps `x-claude-code-session-id` to `trajectory_id`; child-agent requests that include `x-claude-code-agent-id` use that header as the child trajectory and the session ID as the parent.

## OpenCode

OpenCode uses a project-local JSONC provider config; setting an endpoint env var alone is not enough. Create `.opencode/opencode.jsonc` in the project you run OpenCode from:

```jsonc
{
  "provider": {
    "dynamo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Dynamo",
      "env": ["DYNAMO_API_KEY"],
      "models": {
        "zai-org/GLM-4.7-Flash": {
          "id": "zai-org/GLM-4.7-Flash",
          "name": "GLM 4.7 Flash",
          "limit": { "context": 131072, "output": 8192 },
          "cost": { "input": 0, "output": 0 }
        }
      },
      "options": {
        "baseURL": "http://localhost:8000/v1"
      }
    }
  },
  "permission": {
    "task": "allow"
  }
}
```

Run OpenCode with the provider/model pair:

```bash
export DYNAMO_API_KEY=dummy

opencode run -m dynamo/zai-org/GLM-4.7-Flash "Say ok"
```

Dynamo maps OpenCode's `x-session-id` header to `trajectory_id` and `x-parent-session-id` to `parent_trajectory_id`.

## Hermes Agent

This section is intentionally left blank while the Hermes-specific setup settles.

## See Also

- [Trajectory IDs](trajectory-ids.md)
- [Agent Tracing](agent-tracing.md)
- [SGLang for Agentic Workloads](../backends/sglang/agents.md)
