<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Omnigent compatibility assessment

This assessment is pinned to Omnigent commit `733234c303af7254597f99b14bda058878d3e8ca` and `uv 0.11.8`, the minimum version accepted by that checkout. The runner refuses a different or dirty Omnigent checkout so the invocation cannot silently drift from the audited source. It bootstraps the pinned `uv` through `uvx` in the disposable runtime cache and does not update the system installation.

## Supported path

- Headless entry point: `omnigent run --harness codex --model <model> -p <prompt> --no-log`. The no-agent harness form already generates a temporary agent spec and rejects `--no-session`.
- Provider path: a `kind: gateway` provider with an `openai` family, `base_url: <Dynamo>/v1`, `wire_api: responses`, an external `api_key_ref`, and a pinned default model.
- Wire protocol: Codex sends streaming `POST /v1/responses` requests with bearer authentication.
- Session affinity: Codex sends its app-server thread identifier in the `thread-id` header. Dynamo maps that value to normalized `agent_context.session_id` and uses the resulting context for routing; it does not rewrite the HTTP request into an `x-dynamo-session-id` header. A headless run may issue independent main-turn and background requests with different thread IDs; each remains an independently routable Dynamo program. The one-shot capture checks that every Responses request has a non-empty thread ID but does not prove reuse of the main thread across multiple user turns.
- Isolation: the helper gives Omnigent temporary config, data, home, Codex, and uv-cache directories, explicitly runs `omnigent stop` after the turn, verifies the run-owned `.codex-tmp` directory is gone, and only then removes the disposable runtime. It also disables browser launch, telemetry, and update checks. The named API-key variable is mirrored to Omnigent's `OMNIGENT_`-prefixed runner environment so the local host/runner process boundary preserves the external secret reference.

## Lifecycle assessment

Omnigent keeps one Codex app-server thread alive for the wrapped executor session and closes the app-server plus its private `CODEX_HOME` during shutdown. At the audited commit, that shutdown does not send an HTTP request with `x-dynamo-session-final: true` to the model gateway. A successful local capture therefore proves request compatibility and session affinity, but it intentionally reports `lifecycle_qualified: false`.

Use this path for stock Dynamo KV routing and compatibility smoke tests. Do not treat it as ThunderAgent lifecycle-qualified until Omnigent or a narrowly scoped forwarding helper emits a terminal request for the observed Codex thread ID. Process exit alone is not equivalent to Dynamo session finalization.

## Ownership and dependency boundary

| Layer | Owner and responsibility |
| --- | --- |
| Omnigent OSS | Databricks AI and Neon project; owns the local host/runner, temporary agent spec, session/tmux lifecycle, configuration, and wrapped-harness orchestration. |
| Codex CLI | OpenAI project; owns the Responses wire, `thread-id`, turn metadata, tools, and app-server process used by this path. |
| Dynamo | NVIDIA project; owns the OpenAI-compatible endpoint, Codex-header normalization, request tracing, routing, inference, and optional ThunderAgent program state. |
| Integration helper | This experimental branch; owns pin validation, disposable local state, provider projection, credential reference, capture assertions, and explicit `omnigent stop`. |

The audited local path does not require a Databricks workspace, model-serving endpoint, Unity Catalog, or other Databricks-managed runtime. Those may be separate Omnigent deployment options, but they were not needed or qualified here.

## Local evidence — 2026-08-20

A clean capture against the pinned checkout completed with exit code 0. It observed two authenticated, streaming `POST /v1/responses` requests using the expected model: one main turn and one background title turn. Both had non-empty, distinct Codex `thread-id` values and turn metadata. The assistant reply was consumed, `omnigent stop` exited 0, and the run left no `.codex-tmp` or disposable runtime behind. No `x-dynamo-session-final` request was observed.

Classification: experimental stock-Dynamo compatibility path; continue for Kubernetes/basic tool smoke, but defer ThunderAgent qualification until terminal delivery and persistent main-thread reuse are proven. No Kubernetes resource or GPU was used for this local evidence.

## Verification

Run a credential-free local capture against the pinned Omnigent checkout:

```bash
.agents/skills/dynamo-agent-harness/scripts/drive_omnigent.py capture \
  --omnigent-repo /absolute/path/to/omnigent \
  --model capture-model \
  --cwd /absolute/worktree
```

The command succeeds only when the Omnigent subprocess exits successfully, `omnigent stop` succeeds, local Codex temporary state is removed, and the capture observes the Responses wire, expected model, bearer authentication, and a non-empty Codex `thread-id` on every Responses request. The JSON result should report `protocol_compatible: true`, `session_affinity_ok: true`, `persistent_thread_reuse_verified: false`, `session_final_seen: false`, `lifecycle_qualified: false`, `cleanup_exit_code: 0`, and `codex_temp_clean: true`.

Run against a reachable Dynamo endpoint:

```bash
export DYNAMO_API_KEY=dummy
.agents/skills/dynamo-agent-harness/scripts/drive_omnigent.py run \
  --omnigent-repo /absolute/path/to/omnigent \
  --base-url http://127.0.0.1:8000 \
  --model zai-org/GLM-4.7-Flash \
  --cwd /absolute/worktree \
  --prompt "Inspect one file and report one verified fact."
```
