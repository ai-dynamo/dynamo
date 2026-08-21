<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek Harness with Dynamo

This recipe runs DeepSeek Harness (DSH) headless against Dynamo's OpenAI-compatible Chat Completions endpoint. It keeps DSH-native request identity on the model-hidden transport, captures redacted protocol evidence, and can send an explicit terminal signal to the native ThunderAgent router after normal or interrupted shutdown.

## Frozen tuple

| Artifact | Revision |
| --- | --- |
| Dynamo base | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| Dynamo DSH normalization | `1b373f91a451d6a242aafd8851390f8ffdf4c3dc` |
| Published DSH basic client | `@deepseek-ai/dsh@0.1.0-rc.8` |
| DSH source base | `141eb6fef83422698aef7a981029e843e8161534` (`dsh-v0.1.0-rc.8`) |
| DSH full lineage patch | [`0001-feat-llm-propagate-parent-session-lineage.patch`](patches/0001-feat-llm-propagate-parent-session-lineage.patch) |
| Client base image | `node:24.10.0-bookworm-slim@sha256:b8d2197aff9129d16c801a3e3e1b2a873c4946480f5a310f38056df2268c38d9` |

`DSH-basic` uses the published package and proves native root session identity, user-message/tool-result trigger derivation, compaction tagging, and optional terminal cleanup. `DSH-full` applies the included patch so each session-backed child request also carries its immediate durable parent. Keep the patch reviewable as a DSH-owned change until it is accepted upstream; do not hide it in the Codex path.

## DSH-basic: one command

Set `DYNAMO_API_KEY` only when the endpoint requires it. The driver otherwise uses `dummy`, creates an isolated temporary DSH home and generated model profile, binds its relay only to loopback, and deletes the temporary home after the run.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Inspect this workspace with tools, report one verified fact, and do not modify files.' --capture evidence/dsh-basic.jsonl
```

The package is pinned in the driver. A successful trace has a `request` record with `x-deepseek-harness-session-id`, a hashed `x-deepseek-harness-user-id`, the request body used by Dynamo to derive `input_trigger`, and a corresponding `response` status. Compaction requests additionally carry `x-deepseek-harness-compact: 1` and normalize to `agent_context.compaction`.

## DSH-full: child lineage

Build the pinned source plus the included mail patch. `--ignore-scripts` avoids installing repository-owned Git hooks; it does not change the built runtime.

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git /tmp/deepseek-harness-dynamo
git -C /tmp/deepseek-harness-dynamo checkout 141eb6fef83422698aef7a981029e843e8161534
git -C /tmp/deepseek-harness-dynamo am "$PWD/examples/agent_harnesses/deepseek_harness/patches/0001-feat-llm-propagate-parent-session-lineage.patch"
corepack pnpm --dir /tmp/deepseek-harness-dynamo install --frozen-lockfile --ignore-scripts
corepack pnpm --dir /tmp/deepseek-harness-dynamo run build:lib
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Use a DSH subagent to inspect one file, then verify its answer with a local tool.' --dsh-bin /tmp/deepseek-harness-dynamo/apps/cli/lib/bin.js --capture evidence/dsh-full.jsonl
```

For a child request, the full trace preserves both `x-deepseek-harness-session-id` and `x-deepseek-harness-parent-session-id`. Dynamo maps them to `agent_context.session_id` and `agent_context.parent_session_id`; session affinity remains keyed to the child so concurrent children are independently routable.

## ThunderAgent terminal cleanup

Add `--session-final` only when the endpoint is the native ThunderAgent frontend. The relay records every observed DSH session, waits for DSH to flush and exit, then sends one model-hidden `x-dynamo-session-final: true` request per session. It does the same after SIGINT or SIGTERM, bounds each terminal call to five seconds by default, and exits nonzero if Dynamo rejects cleanup.

```bash
node .agents/skills/dynamo-agent-harness/scripts/drive_deepseek_harness.mjs --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B --task 'Run a tool and summarize its result.' --session-final --capture evidence/dsh-thunderagent.jsonl
```

Do not use `--session-final` with a stock KV frontend. Stock KV has no program lifecycle to close, and a generic Chat Completions frontend could treat the terminal envelope as ordinary model work. ThunderAgent consumes it at the router before model forwarding.

## Kubernetes client job

Build and publish the basic client image from the repository root, replace the example image in [`kubernetes/job.yaml`](kubernetes/job.yaml) with the immutable registry digest, and update [`kubernetes/configmap.yaml`](kubernetes/configmap.yaml) with a project-owned Dynamo service and model. The manifests create only a CPU client in `anish-agent-well-lit-path`; they do not deploy or claim a GPU backend.

```bash
docker build -f examples/agent_harnesses/deepseek_harness/Dockerfile.client -t REGISTRY/dsh-dynamo-client:0.1.0-rc.8 .
docker push REGISTRY/dsh-dynamo-client:0.1.0-rc.8
kubectl apply -k examples/agent_harnesses/deepseek_harness/kubernetes
kubectl -n anish-agent-well-lit-path logs -f job/dsh-dynamo-client
kubectl -n anish-agent-well-lit-path cp dsh-dynamo-client-POD:/evidence/dsh-request-trace.jsonl ./dsh-request-trace.jsonl
```

Use [`Dockerfile.full`](Dockerfile.full) for a client image that already contains the pinned lineage patch. Set `DSH_BIN=/opt/deepseek-harness/apps/cli/lib/bin.js` in the Job and use that image's immutable digest. Set `DSH_SESSION_FINAL=true` only for a ThunderAgent service.

Before applying anything on NScale, recalculate cluster allocation, confirm the namespace is still absent or project-owned, and keep all well-lit-path workloads below two nodes and 16 GPUs in aggregate. Never pin a node, copy another workload's DRA claim, tolerate a reserved taint, or mutate node labels/taints.

## Evidence and acceptance

Run the local compatibility checks with:

```bash
node --test .agents/skills/dynamo-agent-harness/scripts/test_drive_deepseek_harness.mjs
cargo test -p dynamo-llm --no-default-features agent_context_from_deepseek_harness_headers_preserves_compaction
```

For a real tool/subagent run, retain the relay JSONL, Dynamo request trace, DSH stdout/stderr, Kubernetes manifests, image digests, model identity, frontend/router logs, and worker telemetry. Match the relay's native session/parent values to Dynamo's normalized `agent_context`; for ThunderAgent, also prove the final request was handled without model forwarding and that no program remains live.

## Security and ownership boundary

The relay forwards the DSH credential but never writes it to evidence. It records model request bodies in plaintext, hashes the stable anonymous DSH user ID, and creates the trace with owner-only permissions; treat the resulting file as sensitive because prompts, tool results, paths, and source excerpts can still be present. DSH owns its profiles, persistence, tools, sandbox, credentials, subagent behavior, and future native lifecycle hooks. Dynamo owns protocol normalization, affinity/routing, request tracing, and ThunderAgent program lifecycle. The relay is an integration shim, not a second DSH runtime or a public Web gateway.

## Known limitations

- The published `DSH-basic` package does not carry parent lineage; use the included full patch when child topology is required.
- The headless product surface is one-shot. Persistent multi-turn qualification requires DSH's ACP/SDK surface or repeated sessions and is not claimed by this recipe.
- The terminal hook observes sessions at the relay. A DSH child that never reaches the model endpoint cannot be discovered or finalized by the shim.
- Input triggers are derived by Dynamo from the Chat Completions body, not emitted as a DSH-specific header. Validate `user_message` and `tool_result` in Dynamo request traces.
- The relay is loopback-only and intentionally minimal; it is not a multi-tenant authentication, rate-limit, or policy boundary.
- A full patched client image is intentionally separate from the published basic package until the lineage change is upstreamed.

## Upstream hook proposal

DSH should expose an optional lifecycle observer at the same boundary that owns a session-backed model request: `onSessionObserved({sessionId,parentSessionId})` and an awaited process-drain callback that receives the durable sessions being closed. A Dynamo plugin can then send `x-dynamo-session-final` without a relay, while stock DSH remains provider-neutral. The hook should be model-hidden, asynchronous, bounded, invoked after session persistence flush, and exercised on normal exit, SIGINT, SIGTERM, and partial startup failure.
