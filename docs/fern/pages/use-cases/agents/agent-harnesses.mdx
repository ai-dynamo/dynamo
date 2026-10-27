---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Connect an Agent Harness
subtitle: Point Claude Code, Codex, OpenCode, and other coding agents at a Dynamo deployment
---

By the end of this guide, your coding agent is sending its requests to Dynamo, and each request carries the agent's own session ID.

Dynamo serves `v1/chat/completions`, `v1/responses`, and `v1/messages`, so any agent that uses one of these APIs can talk to a Dynamo endpoint, even if it is not listed here. The harnesses below also send stable session IDs, which Dynamo normalizes for tracing and other explicitly configured consumers.

## Before You Start

You need a running Dynamo endpoint. To start a local one, follow [Run an agent on Dynamo](run-an-agent-on-dynamo.md). The examples use `http://localhost:8000` and `zai-org/GLM-4.7-Flash`; replace them with your endpoint and model.

## Configure Your Harness

<Tabs>
  <Tab title="Codex">

    Codex uses the Responses API. Add a local provider in `~/.codex/config.toml`:

    ```toml
    [model_providers.dynamo]
    name = "dynamo"
    base_url = "http://localhost:8000/v1"
    wire_api = "responses"
    ```

    Dynamo's Responses adapter currently supports function tools, including functions inside namespaces, and `tool_choice` values of `none`, `auto`, `required`, or a named function. Unsupported tool or tool-choice types return HTTP 400 before inference. Unsupported tools are rejected even when `tool_choice` is `none` or `auto`.

    Disable Codex's hosted web search when using Dynamo. The following command overrides the top-level `web_search` setting for this run:

    ```bash
    # replace -m <model> with your model
    codex -m zai-org/GLM-4.7-Flash -c model_provider=dynamo \
      -c 'web_search="disabled"'
    ```

    Dynamo maps Codex's `thread-id` header to `session_id`. A spawned child thread also sends `x-codex-parent-thread-id`, which Dynamo maps to `parent_session_id`.

  </Tab>
  <Tab title="Pi">

    Pi uses the Dynamo provider plugin. Build and install it from the [agent-plugins](https://github.com/ai-dynamo/agent-plugins/tree/main/pi-plugin) checkout:

    ```bash
    git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
    cd ~/agent-plugins/pi-plugin
    npm install && npm run build
    pi install "$PWD"
    ```

    Point it at the Dynamo OpenAI-compatible endpoint and run Pi with the `dynamo` provider:

    ```bash
    export DYNAMO_BASE_URL=http://localhost:8000/v1
    export DYNAMO_API_KEY=dynamo-local

    pi --model dynamo/zai-org/GLM-4.7-Flash
    ```

  </Tab>
  <Tab title="Claude Code">

    Claude Code uses the Anthropic-compatible Messages API. The local launcher above starts `dynamo.frontend` with `--enable-anthropic-api`; for other deployments, pass that flag when starting the frontend. Then set:

    ```bash
    export ANTHROPIC_BASE_URL=http://localhost:8000
    export ANTHROPIC_MODEL=zai-org/GLM-4.7-Flash
    export ANTHROPIC_SMALL_FAST_MODEL=zai-org/GLM-4.7-Flash
    export CLAUDE_CODE_ATTRIBUTION_HEADER=0 # preserve KV cache hits
    export ANTHROPIC_API_KEY=

    claude
    ```

    Dynamo uses `x-claude-code-session-id` as the Claude Code session ID. For subagents, Dynamo uses `x-claude-code-agent-id` as the child session ID. Nested subagents use `x-claude-code-parent-agent-id` as the parent; top-level subagents fall back to the root session ID.

  </Tab>
  <Tab title="OpenCode">

    OpenCode uses a project-local JSONC provider config; setting an endpoint environment variable alone is not enough. Create `.opencode/opencode.jsonc` in the project where you run OpenCode:

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

    Dynamo maps OpenCode's `x-session-id` header to `session_id` and `x-parent-session-id` to `parent_session_id`.

  </Tab>
  <Tab title="OpenClaw">

    OpenClaw can use Dynamo through its OpenAI-compatible Responses endpoint. Install the Dynamo provider plugin:

    ```bash
    git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
    openclaw plugins install --link ~/agent-plugins/openclaw-plugin
    openclaw plugins enable dynamo
    ```

    Add a Dynamo-backed model to `~/.openclaw/openclaw.json`:

    ```jsonc
    {
      "models": {
        "providers": {
          "dynamo": {
            "baseUrl": "http://localhost:8000/v1",
            "apiKey": "dynamo-local",
            "api": "openai-responses",
            "models": [
              {
                "id": "zai-org/GLM-4.7-Flash",
                "name": "Dynamo GLM 4.7 Flash",
                "reasoning": true,
                "contextWindow": 128000,
                "maxTokens": 8192
              }
            ]
          }
        }
      },
      "agents": {
        "defaults": {
          "model": {
            "primary": "dynamo/zai-org/GLM-4.7-Flash"
          }
        }
      }
    }
    ```

    Run OpenClaw:

    ```bash
    openclaw chat
    ```

    The plugin copies OpenClaw's current `sessionId` into `x-dynamo-session-id` on each request. Native subagents receive their own `session_id`, and the immediate parent is recorded as `parent_session_id`.

  </Tab>
  <Tab title="Hermes Agent">

    Hermes uses an OpenAI-compatible custom endpoint. Configure Hermes with the served model name and Dynamo `/v1` base URL:

    ```yaml
    model:
      default: zai-org/GLM-4.7-Flash
      provider: custom
      base_url: http://localhost:8000/v1
      api_mode: chat_completions
    ```

    If your Dynamo endpoint requires authentication, add `api_key: <token>` to the Hermes model config or set `OPENAI_API_KEY`.

    This configuration lets you run Hermes with the `hermes` command. To send session IDs to Dynamo, install the plugin:

    ```bash
    # clone the plugin
    git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
    # link it to where Hermes typically looks for plugins
    ln -sfnT ~/agent-plugins/hermes-plugin ~/.hermes/plugins/dynamo_session
    hermes plugins enable dynamo_session

    # run Hermes
    hermes
    ```

    The plugin copies the Hermes `session_id` into `x-dynamo-session-id` on each LLM request.

  </Tab>
</Tabs>

## Check That Dynamo Sees the Session

Send a request from the harness, then follow the trace check in [Run an agent on Dynamo](run-an-agent-on-dynamo.md#check-the-trace). Each request should show a session ID. If it does not, the harness is not sending a session header; see [Identify agent sessions](session-ids.mdx#native-agent-headers) for the headers Dynamo recognizes.

## Drive a Harness from Another Agent

Use the repository's `.agents/skills/dynamo-agent-harness` skill when an agent needs to keep a Claude Code, Codex, or OpenCode session open across multiple prompts while Dynamo serves the model. The skill uses Agent Client Protocol (ACP) and injects the provider configuration through the child process environment instead of adding harness configuration files to the delegated worktree.

The skill pins the Claude Code and Codex ACP adapters. If a harness release changes the model, endpoint, session header, authentication, or mode configuration, update this page and the skill in the same change after running a persistent two-turn tool smoke test.

## Tool Schema Validation

On Chat Completions and Responses requests, function tools with explicit `strict: true` receive schema preflight before inference. Invalid schemas now return HTTP 400 where earlier versions could accept them. Responses validates every submitted function, including namespace members and functions excluded by `allowed_tools` or `tool_choice: "none"`. Omitted, `null`, or `false` strictness and omitted or `null` parameters retain existing behavior.

Preflight runs independently of structural-tag activation. Passing preflight does not guarantee complete OpenAI compatibility or backend enforcement. See [Request Validation](../tool-calling-and-reasoning/structural-tag.md#request-validation) for object rules, size budgets, and Dynamo's bounded local-reference support.

## Next

[Identify agent sessions](session-ids.mdx) explains how Dynamo reads these headers and how subagents are linked to their parent.
