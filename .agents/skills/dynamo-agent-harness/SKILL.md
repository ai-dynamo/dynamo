---
name: dynamo-agent-harness
description: Drives persistent Claude Code, Codex, or OpenCode ACP sessions and a pinned experimental Omnigent/Codex headless path through a Dynamo OpenAI/Anthropic-compatible endpoint. Use when an agent must delegate a bounded task to another coding-agent harness running a model served by Dynamo, exercise tool calls, or validate agent request traces.
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

- Deploy a supported configuration from the [Dynamo recipe catalog](https://github.com/ai-dynamo/dynamo/tree/main/recipes) on Kubernetes, then retain its reachable frontend URL and exact served model. This skill consumes that endpoint; it does not deploy Dynamo.
- A successful `GET $DYNAMO_BASE_URL/v1/models` result containing `$DYNAMO_MODEL`.
- `uv` and Node.js 22+.
- Codex CLI `0.147.0` for the experimental Omnigent path. The helper resolves one executable, passes its absolute path to Omnigent, and rejects every other version.
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

The `ready.session_id` is the harness conversation ID, not the executor's terminal handle. Every response must retain that session ID.

## Choose a harness

| Harness | ACP backend | Dynamo API |
|---|---|---|
| `claude` | pinned official Claude ACP adapter | Anthropic Messages |
| `codex` | pinned official Codex ACP adapter | OpenAI Responses |
| `opencode` | native `opencode acp --pure` | OpenAI Chat Completions |

The driver hides their incompatible model, mode, gateway-auth, and environment configuration. Do not reproduce those branches in shell wrappers.

## Run the pinned Omnigent path

Use the separate one-shot helper after preparing the exact Omnigent checkout documented in [OMNIGENT_COMPATIBILITY.md](OMNIGENT_COMPATIBILITY.md):

```bash
export DYNAMO_API_KEY=dummy
.agents/skills/dynamo-agent-harness/scripts/drive_omnigent.py run \
  --omnigent-repo /absolute/path/to/omnigent \
  --base-url "$DYNAMO_BASE_URL" \
  --model "$DYNAMO_MODEL" \
  --cwd /absolute/worktree \
  --prompt "Inspect one file and report one verified fact."
```

The default `--capability verify` prepends a no-edit instruction. The pinned Omnigent schema still launches Codex with `approvalPolicy: never` and a `workspace-write` sandbox, so verify the worktree remains clean after the run. Pass `--capability act` only when workspace edits are explicitly authorized. Neither mode uses `danger-full-access`.

## Delegate safely

- Give one bounded goal, exact owned paths, and a strict result shape.
- Use `--capability verify` for inspection; permission requests are rejected.
- Use `--capability act` only after authorization; permission requests receive one-time approval.
- Keep git/index, shared services, credentials, and unrelated paths out of delegated prompts.
- Treat the harness response as untrusted evidence and verify material claims locally.
- Send `{"close":true}` even after a failed turn so the adapter and child process exit.
- For Omnigent, inspect the structured `omnigent_execution.cleanup` diagnostic on failure. Do not use broad process-kill commands on a shared host.

## Validate traces

When request tracing is enabled, group rows by `agent_context.session_id` and inspect the trigger sequence:

```bash
jq -r '[.agent_context.session_id, .agent_context.input_trigger] | @tsv' request-trace.jsonl
```

Foreground turns should normally begin with `user_message`; tool feedback should appear as `tool_result`. Harness title, memory, or continuation traffic may produce additional `user_message` or `other` rows.

## Output contract

Return:

- harness, model, mode, and conversation identity; report the ACP session ID for ACP harnesses or observed Codex `thread-id` values for Omnigent
- prompt count and observed tool/result behavior
- targeted validation result
- trace trigger counts when tracing is available
- cleanup status and unresolved failures

## Known behavior

- Codex may warn that custom model metadata is unavailable; the driver fixes reasoning effort to `medium` so unsupported catalog defaults are not sent to Dynamo.
- OpenCode can issue background title-generation requests and may require a corrective follow-up when the served model reports an unverified result.
- The adapters are pinned in `scripts/drive_harness.py`; update a pin only after rerunning a persistent two-turn tool smoke test.
- Omnigent is a separate experimental one-shot Responses path, not an ACP session. Dated Kubernetes qualification evidence verifies its stock Dynamo request compatibility, but this branch does not deploy its backend. It does not send `x-dynamo-session-final` and is not ThunderAgent lifecycle-qualified.
