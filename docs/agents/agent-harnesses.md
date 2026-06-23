---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Agent Harnesses
subtitle: Point coding-agent CLIs at a Dynamo deployment
---

Dynamo exposes `v1/chat/completions`, `v1/responses`, and `v1/messages` so **any agent** that uses these APIs can talk to a Dynamo endpoint even if its not explicitly listed in this guide. This guide specifically focuses on popular agent harnesses that send stable trajectory ID that Dynamo can use for optimized routing and scheduling.

## Local Setup

To locally test these out, we have a small script that runs an SGLang-backed `zai-org/GLM-4.7-Flash` endpoint. This script starts a TP2 instance on port 8000 and enables request tracing for replay and visualization. By default traces are saved in `/tmp/dynamo-request-trace-$(date +%Y%m%d-%H%M%S)`

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
```

```bash
# replace -m <model> with your model
codex -m zai-org/GLM-4.7-Flash -c model_provider=dynamo
```

Codex sends a `session-id` header that is internally mapped to our `trajectory_id`

## Claude Code

Claude Code uses Anthropic-compatible Messages API. The local launcher above starts `dynamo.frontend` with `--enable-anthropic-api`; for other deployments, pass that flag when starting the frontend. Then set:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_MODEL=zai-org/GLM-4.7-Flash
export ANTHROPIC_SMALL_FAST_MODEL=zai-org/GLM-4.7-Flash
export ANTHROPIC_BASE_URL=http://localhost:8000
export CLAUDE_CODE_ATTRIBUTION_HEADER=0 # preserve kv cache hits!
export ANTHROPIC_API_KEY=

claude
```

Dynamo uses `x-claude-code-session-id` as the Claude Code trajectory ID. For subagents, Dynamo uses `x-claude-code-agent-id` as the child trajectory ID and the session ID as its parent.

## OpenCode

OpenCode uses a project-local JSONC provider config; setting an endpoint env var alone is not enough. Create `.opencode/opencode.jsonc` in the project you run OpenCode from:

```jsonc
{
  "provider": {
    "dynamo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Dynamo",
      "models": {
        "zai-org/GLM-4.7-Flash": {
          "id": "zai-org/GLM-4.7-Flash",
          "name": "GLM 4.7 Flash"
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
opencode -m dynamo/zai-org/GLM-4.7-Flash
```

Dynamo maps OpenCode's `x-session-id` header to `trajectory_id` and `x-parent-session-id` to `parent_trajectory_id`.

## Hermes Agent

Hermes uses an OpenAI-compatible custom endpoint. Configure Hermes with the served model name and Dynamo `/v1` base URL:

```yaml
model:
  default: zai-org/GLM-4.7-Flash
  provider: custom
  base_url: http://localhost:8000/v1
  api_mode: chat_completions
```

If your Dynamo endpoint requires auth, add `api_key: <token>` to the Hermes model config or set `OPENAI_API_KEY`.

Just setting that will allow you to run hermes agent via `hermes`. To enable trajectory ids for Dynamo, you can install the plugin using the instructions below.

```bash
# clone the plugin
git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
# link it to where hermes typically looks for plugins
ln -sfnT ~/agent-plugins/hermes-plugin ~/.hermes/plugins/dynamo_trajectory
hermes plugins enable dynamo_trajectory

# run hermes
hermes
```

The plugin copies the Hermes `session_id` into `x-dynamo-trajectory-id` on each LLM request.

## See Also

- [Trajectory IDs](trajectory-ids.md)
- [Agent Tracing](agent-tracing.md)
- [SGLang for Agentic Workloads](../backends/sglang/agents.md)
