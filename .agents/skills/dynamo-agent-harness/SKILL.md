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

Treat the [Agent Harnesses guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-harnesses.mdx) as the source of truth for harness configuration. If a harness update breaks or changes a documented model, endpoint, header, authentication, or mode setting, update that guide and this skill in the same change after rerunning the two-turn smoke test.

## Prerequisites

- Deploy a supported configuration from the [Dynamo recipe catalog](https://github.com/ai-dynamo/dynamo/tree/main/recipes) on Kubernetes, then retain its reachable frontend URL and exact served model. This skill consumes that endpoint; it does not deploy Dynamo.
- A successful `GET $DYNAMO_BASE_URL/v1/models` result containing `$DYNAMO_MODEL`.
- `uv` and Node.js 22+.
- `opencode` on `PATH` only when selecting the OpenCode harness.
- A working directory that limits the delegated agent's scope.
- `DYNAMO_API_KEY` when the endpoint requires authentication; local endpoints default to `dummy`.

## Start a session

Default to `verify`. Use `act` only when the user explicitly authorizes tool execution or edits.

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000
export DYNAMO_MODEL=your-recipe-served-model

.agents/skills/dynamo-agent-harness/scripts/drive_harness.py \
  --harness codex \
  --base-url "$DYNAMO_BASE_URL" \
  --model "$DYNAMO_MODEL" \
  --cwd /absolute/worktree \
  --capability verify
```

Run the command with a TTY so stdin stays open. Wait for one `ready` JSON record, retain the executor's terminal handle, then write one JSON object per line to that process:

```json
{"prompt":"Inspect src/router.rs. Use tools to test the highest-risk invariant. Do not edit files."}
{"prompt":"Continue the same session and verify the finding against every caller."}
{"close":true}
```

The `ready.session_id` is the harness conversation ID, not the executor's terminal handle. Every response must retain that session ID. This stock-Dynamo command closes the ACP process without sending a lifecycle envelope.

## Finalize a ThunderAgent session

> [!WARNING]
> `--session-final` is a ThunderAgent-only lifecycle operation. Do not use it with a stock Dynamo frontend or KV router: a frontend that does not recognize `x-dynamo-session-final` can treat the terminal envelope as ordinary model work.

For an endpoint that is explicitly deployed with ThunderAgent, add the lifecycle flag:

```bash
.agents/skills/dynamo-agent-harness/scripts/drive_harness.py \
  --harness codex \
  --base-url "$DYNAMO_BASE_URL" \
  --model "$DYNAMO_MODEL" \
  --cwd /absolute/worktree \
  --capability verify \
  --session-final
```

The Codex ACP adapter returns the Codex thread ID as `ready.session_id`. After the ACP process has closed, the driver attempts one terminal request with that exact value as `x-dynamo-session-id` and `x-dynamo-session-final: true`. It performs this cleanup after normal close, a failed turn, or interruption, but only if ACP created a nonempty session ID. ThunderAgent consumes the terminal request without forwarding work to the model. A finalization failure emits a failed `session_final` record and makes an otherwise successful run fail closed; if the session already failed, its original error remains primary and carries a note about the finalization failure.

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
- Leave `--session-final` off for stock Dynamo and KV-router runs because they have no ThunderAgent program lifecycle to terminate.
- The driver gives the child only `PATH`, `HOME`, temporary-directory, runtime-directory, and locale variables; the selected Dynamo API-key variable; and fresh harness-specific endpoint/model configuration. It does not copy other parent credentials or an ambient `CODEX_CONFIG`.

For a complete Codex qualification procedure, evidence checklist, frozen dependency tuple, and cleanup steps, use [`examples/agent_harnesses/codex/README.md`](https://github.com/ai-dynamo/dynamo/blob/main/examples/agent_harnesses/codex/README.md).

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
