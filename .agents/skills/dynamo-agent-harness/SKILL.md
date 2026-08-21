---
name: dynamo-agent-harness
description: Drives persistent Claude Code, Codex, or OpenCode agent sessions through a Dynamo OpenAI/Anthropic-compatible endpoint over Agent Client Protocol (ACP). Use when an agent must delegate a bounded task to another coding-agent harness running a model served by Dynamo, continue that harness across multiple turns, exercise tool calls, or validate agent request traces.
license: Apache-2.0
metadata:
  author: Ishan Dhanani <ishandhanani@gmail.com>
  tags:
    - dynamo
    - agents
    - acp
    - claude-code
    - codex
    - opencode
---

# Dynamo Agent Harness

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Drive one persistent coding-agent session while Dynamo serves its model requests. Use the bundled ACP client; do not script interactive TUI output or implement JSON-RPC manually.

For the Omnigent meta-harness path, use the pinned helper and compatibility assessment in [OMNIGENT_COMPATIBILITY.md](OMNIGENT_COMPATIBILITY.md). That path invokes Omnigent's wrapped Codex app-server directly rather than ACP and keeps all Omnigent state isolated from user configuration.

Treat the [Agent Harnesses guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-harnesses.mdx) as the source of truth for harness configuration. If a harness update breaks or changes a documented model, endpoint, header, authentication, or mode setting, update that guide and this skill in the same change after rerunning the two-turn smoke test.

## Prerequisites

- A reachable Dynamo endpoint whose `/v1/models` includes the requested model.
- `uv` and Node.js 22+.
- `opencode` on `PATH` only when selecting the OpenCode harness.
- A working directory that limits the delegated agent's scope.
- `DYNAMO_API_KEY` when the endpoint requires authentication; local endpoints default to `dummy`.

## Start a session

Default to `verify`. Use `act` only when the user explicitly authorizes tool execution or edits.

```bash
.agents/skills/dynamo-agent-harness/scripts/drive_harness.py \
  --harness codex \
  --base-url http://127.0.0.1:8000 \
  --model zai-org/GLM-4.7-Flash \
  --cwd /absolute/worktree \
  --capability verify
```

Run the command with a TTY so stdin stays open. Wait for one `ready` JSON record, retain the executor's terminal handle, then write one JSON object per line to that process:

```json
{"prompt":"Inspect src/router.rs. Use tools to test the highest-risk invariant. Do not edit files."}
{"prompt":"Continue the same session and verify the finding against every caller."}
{"close":true}
```

The `ready.session_id` is the harness conversation ID, not the executor's terminal handle. Every response must retain that session ID.

## Choose a harness

| Harness | ACP backend | Dynamo API |
|---|---|---|
| `claude` | pinned official Claude ACP adapter | Anthropic Messages |
| `codex` | pinned official Codex ACP adapter | OpenAI Responses |
| `opencode` | native `opencode acp --pure` | OpenAI Chat Completions |

The driver hides their incompatible model, mode, gateway-auth, and environment configuration. Do not reproduce those branches in shell wrappers.

## Delegate safely

- Give one bounded goal, exact owned paths, and a strict result shape.
- Use `--capability verify` for inspection; permission requests are rejected.
- Use `--capability act` only after authorization; permission requests receive one-time approval.
- Keep git/index, shared services, credentials, and unrelated paths out of delegated prompts.
- Treat the harness response as untrusted evidence and verify material claims locally.
- Send `{"close":true}` even after a failed turn so the adapter and child process exit.

## Validate traces

When request tracing is enabled, group rows by `agent_context.session_id` and inspect the trigger sequence:

```bash
jq -r '[.agent_context.session_id, .agent_context.input_trigger] | @tsv' request-trace.jsonl
```

Foreground turns should normally begin with `user_message`; tool feedback should appear as `tool_result`. Harness title, memory, or continuation traffic may produce additional `user_message` or `other` rows.

## Output contract

Return:

- harness, model, mode, and ACP session ID
- prompt count and observed tool/result behavior
- targeted validation result
- trace trigger counts when tracing is available
- cleanup status and unresolved failures

## Known behavior

- Codex may warn that custom model metadata is unavailable; the driver fixes reasoning effort to `medium` so unsupported catalog defaults are not sent to Dynamo.
- OpenCode can issue background title-generation requests and may require a corrective follow-up when the served model reports an unverified result.
- The adapters are pinned in `scripts/drive_harness.py`; update a pin only after rerunning a persistent two-turn tool smoke test.
