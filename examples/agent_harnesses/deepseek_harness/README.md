<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek Harness with Dynamo

This package-only recipe runs the published DeepSeek Harness (DSH) headless profile against an existing Dynamo OpenAI-compatible Chat Completions endpoint. It preserves DSH-native root-session and compaction metadata on the model-hidden transport, captures redacted protocol evidence, and can send an explicit terminal signal to the native ThunderAgent router after normal or interrupted shutdown. It does not deploy Dynamo or prescribe a Kubernetes topology.

## Frozen tuple

| Artifact | Revision |
| --- | --- |
| Dynamo base | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| Dynamo DSH normalization | `1b373f91a451d6a242aafd8851390f8ffdf4c3dc` plus the package-only cleanup in this branch |
| Complete recipe revision | `org.opencontainers.image.revision` image label, set from the clean checkout commit at build time |
| Published DSH client | `@deepseek-ai/dsh@0.1.0-rc.8` |
| Client base image | `node:24.10.0-bookworm-slim@sha256:b8d2197aff9129d16c801a3e3e1b2a873c4946480f5a310f38056df2268c38d9` |
| Image dependency graph | [`client/pnpm-lock.yaml`](client/pnpm-lock.yaml), installed by pnpm `11.7.0` with `--frozen-lockfile` |

No DSH source checkout or source patch is part of this supported path. Immediate parent lineage is not an acceptance criterion because the published package does not emit it.

## Deploy Dynamo first

Install the [Dynamo Kubernetes Platform](../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx), then choose and deploy a supported configuration from the [Dynamo recipe catalog](../../../recipes/README.md). Follow that recipe through model preparation, `DynamoGraphDeployment` readiness, and frontend exposure. The harness path begins only after the recipe has produced a reachable frontend URL and served model name.

Export those two outputs and verify the model before starting DSH. `DYNAMO_BASE_URL` is the frontend origin without a required `/v1` suffix; the driver normalizes it.

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000
export DYNAMO_MODEL=your-recipe-served-model
curl -fsS "$DYNAMO_BASE_URL/v1/models" | jq -e --arg model "$DYNAMO_MODEL" '.data[]? | select(.id == $model)'
```

The remaining commands assume the recipe deployment stays running and these variables identify its endpoint. If the recipe requires authentication, export `DYNAMO_API_KEY` as well.

## Run one headless task

Run the driver on a POSIX host with Linux `/proc` or `/bin/ps`. Set `DYNAMO_API_KEY` only when the endpoint requires it. The driver never reads an ambient `DEEPSEEK_API_KEY`; it selects `DYNAMO_API_KEY`, projects that value into the variable expected by DSH, passes a small allowlist of noncredential environment variables, and otherwise uses `dummy`. Use `--api-key-env NAME` only when explicitly selecting another credential variable. The driver creates an isolated temporary home and generated model profile, binds its relay only to loopback, and deletes the temporary home after the run.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url "$DYNAMO_BASE_URL" --model "$DYNAMO_MODEL" --task 'Inspect this workspace with tools, report one verified fact, and do not modify files.' --capture evidence/dsh-headless.jsonl
```

The capture path uses exclusive creation. Choose a new path for each run. Add `--overwrite-capture` only when replacing that exact evidence file is intentional.

The package is pinned in the driver. A successful trace has a `request` record with `x-deepseek-harness-session-id`, a hashed `x-deepseek-harness-user-id`, the request body used by Dynamo to derive `input_trigger`, and a corresponding `response` status. Compaction requests additionally carry exact `x-deepseek-harness-compact: 1` and normalize to `agent_context.compaction`.

## Why this recipe is one-shot

This is a choice of DSH surface, not a Dynamo session limitation. The published `dsh --profile headless "task"` command is explicitly implemented as a one-task application: it creates one fresh persisted Agent, submits one top-level task, waits until the Agent is idle, flushes the session, prints the final assistant message, requests application exit, and terminates the process.

One top-level task can still produce several model calls, tool-result turns, compaction requests, and subagent activity. Those requests retain stable DSH session identity for the lifetime of the task. “One-shot” means the host process accepts no second top-level prompt after the answer; it does not mean one HTTP request or an ephemeral model context.

Keeping `DSH_HOME` preserves session artifacts but does not turn the headless command into an interactive server. A persistent automation path should be a separate recipe built from the published `@deepseek-ai/dsh-acp` package: keep one ACP stdio connection alive, call `session/new` once, then send repeated `session/prompt` requests for that session. DSH ACP currently owns sessions for the life of the connection and supports repeated prompts, but does not support loading or resuming a session after the connection exits. Human-interactive persistence belongs to DSH's TUI or Web profiles, not this headless process.

## Legacy Dynamo compatibility bridge

Add `--canonicalize-dynamo-headers` only when the selected recipe intentionally runs a legacy Dynamo release that predates native DSH normalization. The relay preserves the native DSH header and also copies its session value to `x-dynamo-session-id`. This restores identity and affinity on that older server, but it cannot add native DSH compaction normalization. Leave the option off for current deployments so the native mapping remains the contract under test.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url "$DYNAMO_BASE_URL" --model "$DYNAMO_MODEL" --task 'Inspect this workspace with tools and report one verified fact.' --canonicalize-dynamo-headers --capture evidence/dsh-legacy-dynamo.jsonl
```

## ThunderAgent terminal cleanup

Add `--session-final` only when the endpoint is the native ThunderAgent frontend. The relay records every observed DSH session, waits for DSH to flush and exit, then sends one model-hidden `x-dynamo-session-final: true` request per session. It does the same after SIGINT or SIGTERM, bounds each terminal call to five seconds by default, and exits nonzero if Dynamo rejects cleanup or if no DSH session reached Dynamo. SIGINT and SIGTERM terminate the complete tracked Corepack/pnpm/DSH process tree, including detached descendants observed before shutdown, and retain conventional exit codes `130` and `143` after the bounded drain.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url "$DYNAMO_BASE_URL" --model "$DYNAMO_MODEL" --task 'Run a tool and summarize its result.' --session-final --capture evidence/dsh-thunderagent.jsonl
```

Do not use `--session-final` with a stock KV frontend. Stock KV has no program lifecycle to close, and a generic Chat Completions frontend could treat the terminal envelope as ordinary model work. ThunderAgent consumes it at the router before model forwarding.

## Evidence and acceptance

Run the local compatibility checks with:

```bash
DSH_PACKAGE_SMOKE=1 node --test .agents/skills/dynamo-agent-harness/scripts/test_drive_deepseek_harness.mjs
cargo test -p dynamo-llm --no-default-features agent_context_from_deepseek_harness_headers_preserves_compaction
```

For a real tool run, retain the relay JSONL, Dynamo request trace, the exact deployed recipe revision and manifest, DSH stdout and stderr, model identity, frontend or router logs, and worker telemetry. Match the relay's native session value to Dynamo's normalized `agent_context.session_id`; for ThunderAgent, also prove the final request was handled without model forwarding and that no program remains live.

## Security and ownership boundary

The relay reads only `DYNAMO_API_KEY` by default, projects the selected value to DSH as `DEEPSEEK_API_KEY`, and never writes it to evidence. The DSH process receives an isolated home plus an allowlist containing executable search path, locale, terminal, temporary-directory, timezone, and certificate settings; unrelated parent credentials and package-manager cache locations are not inherited. The relay records model request bodies in plaintext, hashes the stable anonymous DSH user ID, and creates the trace with owner-only permissions and exclusive creation by default. Treat the resulting file as sensitive because prompts, tool results, paths, and source excerpts can still be present. DSH owns its profiles, persistence, tools, sandbox, credentials, subagent behavior, and future native lifecycle hooks. Dynamo owns protocol normalization, affinity and routing, request tracing, and ThunderAgent program lifecycle. The relay is an integration shim, not a second DSH runtime or a public Web gateway.

## Known limitations

- This supported path carries root-session and compaction metadata only; immediate parent lineage is intentionally out of scope.
- A direct local run uses a top-level-pinned `pnpm dlx` package and can resolve transitive dependencies again. Use the container's frozen lockfile for reproducible qualification evidence.
- The headless product surface accepts one top-level task. Persistent multi-turn DSH requires a separately packaged ACP host and is not claimed by this recipe.
- The terminal hook observes sessions at the relay. A DSH child that never reaches the model endpoint cannot be discovered or finalized by the shim.
- Input triggers are derived by Dynamo from the Chat Completions body, not emitted as a DSH-specific header. Validate `user_message` and `tool_result` in Dynamo request traces.
- The relay is loopback-only and intentionally minimal; it is not a multi-tenant authentication, rate-limit, or policy boundary.
- The Dynamo 1.3 compatibility bridge preserves identity and affinity but cannot backport native DSH compaction normalization.
- Windows is not qualified because the relay requires POSIX process-tree signaling.

## Future persistent path

The next DSH milestone should compose the published `@deepseek-ai/dsh-acp` server with the same pinned model, sandbox, and persistence plugins, then add an ACP host driver that keeps one connection open for multiple prompts and translates connection teardown into bounded ThunderAgent finalization. This is integration and packaging work in Dynamo; it does not require a DSH source patch.
