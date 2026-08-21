---
name: dynamo-agent-harness
description: Drives persistent Claude Code, Codex, or OpenCode ACP sessions and one-shot DeepSeek Harness sessions through a Dynamo OpenAI/Anthropic-compatible endpoint. Use when an agent must delegate a bounded task to a coding-agent harness running a model served by Dynamo, exercise tool calls, validate agent request traces, or prove lifecycle cleanup.
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

Drive a coding-agent session while Dynamo serves its model requests. Use the bundled ACP client for Claude Code, Codex, and OpenCode. Use the dedicated headless relay for DeepSeek Harness (DSH); do not script interactive TUI output or implement JSON-RPC manually.

Treat the [Agent Harnesses guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-harnesses.mdx) as the source of truth for harness configuration. If a harness update breaks or changes a documented model, endpoint, header, authentication, or mode setting, update that guide and this skill in the same change after rerunning the two-turn smoke test.

## Prerequisites

- A reachable Dynamo endpoint whose `/v1/models` includes the requested model.
- `uv` and Node.js 22+.
- Node.js 24+ for DSH.
- A POSIX host with Linux `/proc` or `/bin/ps` for process-tree cleanup.
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

## Run DeepSeek Harness

DSH is not driven through the ACP script above. Its shipped `headless` profile creates one fresh persisted session, runs one task, flushes it, prints the final answer, and exits. The dedicated relay pins `@deepseek-ai/dsh@0.1.0-rc.8`, generates an isolated model profile, preserves native request headers, passes a minimal child environment, and writes redacted JSONL evidence with exclusive creation:

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Use tools to inspect this workspace and report one verified fact.' --capture dsh-request-trace.jsonl
```

The relay reads `DYNAMO_API_KEY` only, then projects the selected value to the variable expected by DSH. An ambient `DEEPSEEK_API_KEY` is ignored. Use `--api-key-env NAME` only to explicitly select a different credential variable, and use `--overwrite-capture` only to intentionally replace an existing trace.

Use `--canonicalize-dynamo-headers` only with an older server such as the pinned Dynamo 1.3 deployment. It preserves native headers while copying DSH session and parent identity into canonical Dynamo headers; it cannot backport native DSH compaction normalization. Leave it off against a server that includes the native mapping.

Use `--session-final` only against Dynamo's native ThunderAgent frontend. It sends one canonical final request for every DSH session observed by the relay after normal exit, SIGINT, or SIGTERM, and fails closed when cleanup is rejected or zero sessions were observed. Signals terminate the complete tracked Corepack/pnpm/DSH process tree, including detached descendants observed before shutdown, and return `130` for SIGINT or `143` for SIGTERM after cleanup. Do not enable it for stock KV endpoints. Use the full pinned lineage recipe in [`examples/agent_harnesses/deepseek_harness`](../../../examples/agent_harnesses/deepseek_harness/README.md) when a child must carry `x-deepseek-harness-parent-session-id`.

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

- harness, model, mode, and harness session ID; call it an ACP session ID only for ACP-driven harnesses
- prompt count and observed tool/result behavior
- targeted validation result
- trace trigger counts when tracing is available
- cleanup status and unresolved failures

## Known behavior

- Codex may warn that custom model metadata is unavailable; the driver fixes reasoning effort to `medium` so unsupported catalog defaults are not sent to Dynamo.
- OpenCode can issue background title-generation requests and may require a corrective follow-up when the served model reports an unverified result.
- The adapters are pinned in `scripts/drive_harness.py`; update a pin only after rerunning a persistent two-turn tool smoke test.
- DSH basic is pinned in `scripts/drive_deepseek_harness.mjs`. Its published package carries native session and compaction headers but needs the separately reviewable full patch for parent lineage.
- DSH's basic container installs its complete dependency graph from the frozen recipe lockfile. A direct local `pnpm dlx` run pins the top-level package only, so use the container or full source build for reproducible qualification evidence.
- DSH request evidence contains plaintext model bodies even though credentials are redacted and its stable anonymous user ID is hashed. Handle the JSONL as sensitive data.
