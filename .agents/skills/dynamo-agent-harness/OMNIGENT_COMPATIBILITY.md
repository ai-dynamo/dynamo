<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Omnigent compatibility assessment

This experimental path runs one pinned Omnigent/Codex turn through a stock NVIDIA Dynamo Responses endpoint. It does not qualify Kubernetes deployment, persistent multi-turn reuse, or ThunderAgent lifecycle cleanup.

## Reproducibility tuple

| Component | Audited identity |
| --- | --- |
| Omnigent | `omnigent-ai/omnigent` commit `733234c303af7254597f99b14bda058878d3e8ca` |
| uv used inside the disposable runtime | `0.11.8` |
| Codex CLI | Exact version `0.147.0` |
| Dynamo integration baseline | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| Dynamo integration branch | `omnigent-well-lit-path`; record the exact `git rev-parse HEAD` used for each build or evidence bundle |

The runner rejects a different or dirty Omnigent checkout. It resolves either `--codex-bin` or one `codex` executable from `PATH`, probes `codex --version`, passes the resolved absolute path through `OMNIGENT_CODEX_PATH`, and fails unless the result is exactly `codex-cli 0.147.0`. It bootstraps pinned `uv 0.11.8` through `uvx` inside the disposable runtime cache and does not update the system installation.

Prepare the audited Omnigent checkout:

```bash
git clone https://github.com/omnigent-ai/omnigent.git /absolute/path/to/omnigent
git -C /absolute/path/to/omnigent checkout --detach 733234c303af7254597f99b14bda058878d3e8ca
git -C /absolute/path/to/omnigent status --short
codex --version
```

The status command must print nothing, and the version command must print `codex-cli 0.147.0`. If several Codex installations exist, pass the intended executable with `--codex-bin /absolute/path/to/codex`.

## Supported path

- Headless entry point: `omnigent run <generated-agent-directory> -p <prompt> --server local --harness codex --model <model> --no-log`. The helper writes an invocation-owned bundle whose `config.yaml` pins Codex, the model, `skills: none`, the workspace, and a hard sandbox. Explicit local-server, harness, and model options prevent configured defaults from redirecting the run. Do not substitute bare `omnigent run --harness codex` at the audited commit: that shorthand generates `sandbox.type: none`.
- Provider path: a `kind: gateway` provider with an `openai` family, `base_url: <Dynamo>/v1`, `wire_api: responses`, an external `api_key_ref`, and a pinned default model.
- Wire protocol: Codex sends streaming `POST /v1/responses` requests with bearer authentication.
- Session affinity: Codex sends its app-server thread identifier in the `thread-id` header. Dynamo maps that value to normalized `agent_context.session_id` and uses the resulting context for routing; it does not rewrite the HTTP request into an `x-dynamo-session-id` header. A headless run may issue independent main-turn and background requests with different thread IDs; each remains an independently routable Dynamo program. The one-shot capture checks that every Responses request has a non-empty thread ID but does not prove reuse of the main thread across multiple user turns.
- Credential boundary: the child environment starts from an explicit allowlist of process settings and contains no ambient GitHub, cloud, Kubernetes, registry, model-provider, or corporate credentials. Only `DYNAMO_API_KEY` is selected from the parent environment; the same value is mirrored to `OMNIGENT_DYNAMO_API_KEY` because Omnigent requires that prefix across its host/runner boundary. An absent or empty value becomes the synthetic local value `dummy`.
- Sandbox: the authored bundle carries the pinned Omnigent `OSEnvSpec` schema with `type: caller_process`, the selected workspace plus invocation-private Codex/temp directories as its write paths, no environment passthrough, and the platform hard sandbox (`darwin_seatbelt` on macOS or `linux_bwrap` on Linux). Linux fails closed when `bwrap` is unavailable. The pinned Codex executor derives `sandbox: workspace-write` and hard-codes `approvalPolicy: never`; the helper never requests `danger-full-access`.
- Capability: `--capability verify` is the default and prepends a no-edit instruction. Because the requested Codex sandbox is still `workspace-write`, that instruction is not a kernel-enforced read-only boundary; check `git status --short` after verification. Pass `--capability act` only when edits inside `--cwd` are authorized.
- Isolation and teardown: the helper gives Omnigent temporary config, data, home, Codex, temp, and uv-cache directories outside the selected workspace, launches from that private runtime so workspace-local `.omnigent/config.yaml` cannot participate, and disables browser launch, telemetry, web search, host skills, and update checks. After success, failure, timeout, or keyboard interruption, pinned internal cleanup reads only the invocation's isolated daemon records and server PID file; it never performs Omnigent's machine-wide canonical-port sweep. The helper rejects a pre-existing workspace `.codex-tmp`, removes run-created empty state after a bounded drain, and then checks that the disposable runtime was removed.

## Lifecycle assessment

Omnigent keeps one Codex app-server thread alive for the wrapped executor session and closes the app-server plus its private `CODEX_HOME` during shutdown. At the audited commit, that shutdown does not send an HTTP request with `x-dynamo-session-final: true` to the model gateway. A successful local capture therefore proves request compatibility and session affinity, but it intentionally reports `lifecycle_qualified: false`.

Use this path for stock Dynamo KV routing and one-shot compatibility smoke tests. Do not treat it as ThunderAgent lifecycle-qualified until Omnigent or a narrowly scoped forwarding helper emits a terminal request for the observed Codex thread ID. Process exit alone is not equivalent to Dynamo session finalization.

## Ownership and dependency boundary

| Layer | Owner and responsibility |
| --- | --- |
| Omnigent OSS | Databricks AI and Neon project; owns the local host/runner, temporary agent spec, session/tmux lifecycle, configuration, and wrapped-harness orchestration. |
| Codex CLI | OpenAI project; owns the Responses wire, `thread-id`, turn metadata, tools, and app-server process used by this path. |
| Dynamo | NVIDIA project; owns the OpenAI-compatible endpoint, Codex-header normalization, request tracing, routing, inference, and optional ThunderAgent program state. |
| Integration helper | This experimental branch; owns pin validation, disposable local state, provider projection, credential reference, capture assertions, and invocation-scoped daemon/server cleanup. |

The audited local path does not require a Databricks workspace, model-serving endpoint, Unity Catalog, or other Databricks-managed runtime. Those may be separate Omnigent deployment options, but they were not needed or qualified here.

## Local evidence — 2026-08-20

A clean post-hardening capture against Omnigent commit `733234c303af7254597f99b14bda058878d3e8ca` and Codex CLI `0.147.0` completed with exit code 0. It observed two authenticated, streaming `POST /v1/responses` requests using the expected model: one main turn and one background title turn. Both had non-empty, distinct Codex `thread-id` values and `seatbelt` turn metadata, proving that the authored `darwin_seatbelt` bundle replaced the shorthand CLI's unsafe `none` sandbox. The assistant reply was consumed, invocation-scoped cleanup stopped one private daemon and the private local server, and the run left no `.codex-tmp` or disposable runtime behind. A final run through the one-GPU NScale stock backend (`Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, 32,768-token context) returned exactly `OMNIGENT_STOCK_OK` and exited 0 under the same scoped cleanup implementation. No `x-dynamo-session-final` request was observed. The environment allowlist, exact Codex probe, authored agent bundle, active-sandbox assertion, assistant-reply requirement, pre-existing-state rejection, and invocation-scoped teardown are covered by 20 deterministic tests.

Classification: experimental stock-Dynamo compatibility path. The shared NScale deployment, not this branch, supplied the GPU backend. ThunderAgent finalization and persistent main-thread reuse are not qualified.

## Verification

Run a credential-free local capture against the pinned Omnigent checkout:

```bash
.agents/skills/dynamo-agent-harness/scripts/drive_omnigent.py capture \
  --omnigent-repo /absolute/path/to/omnigent \
  --model capture-model \
  --cwd /absolute/worktree
```

The command succeeds only when the Omnigent subprocess exits successfully, invocation-scoped cleanup succeeds, local Codex temporary state is removed, the expected assistant reply is consumed, and the capture observes the Responses wire, expected model, bearer authentication, a non-empty Codex `thread-id`, and an active Codex sandbox label on every Responses request. Codex reports abstract policy names such as `read-only` or `workspace-write` when it owns the sandbox and concrete backend names such as `seatbelt`, `bwrap`, or `landlock` when it detects an external platform sandbox; `none`, `danger-full-access`, missing, and unknown labels fail closed. The JSON result should report `protocol_compatible: true`, `safe_sandbox_observed: true`, `assistant_reply_seen: true`, `session_affinity_ok: true`, `persistent_thread_reuse_verified: false`, `session_final_seen: false`, `lifecycle_qualified: false`, `cleanup_exit_code: 0`, and `codex_temp_clean: true`.

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

The command defaults to `--capability verify`. Add `--capability act` only when the task may edit the selected worktree. To pin a non-default executable explicitly, add `--codex-bin /absolute/path/to/codex`.

## Teardown and troubleshooting

The helper's cleanup command imports the pinned Omnigent checkout and terminates only daemon records plus the server PID file under that invocation's private `OMNIGENT_DATA_DIR`. Do not replace it with `omnigent stop`: at the audited commit, that command intentionally sweeps machine-wide daemon records and the canonical server port. On failure, the runner prints `omnigent_execution` JSON to stderr with command exit status, timeout/error state, cleanup exit status, redacted cleanup stdout/stderr, `.codex-tmp` status, and disposable-runtime status. Use that invocation-scoped record; do not use `pkill`, kill by process name, or another broad cleanup command on a shared host.

- If the runner reports a Codex version mismatch, install `codex-cli 0.147.0` or pass the exact executable with `--codex-bin`.
- If Linux reports that `bwrap` is missing, install bubblewrap. The helper intentionally has no unsandboxed fallback.
- If Dynamo returns `401` or `403`, export only the endpoint's `DYNAMO_API_KEY`; do not reuse `OPENAI_API_KEY` or another provider credential.
- If the Omnigent checkout validation fails, restore the checkout to detached commit `733234c303af7254597f99b14bda058878d3e8ca` and remove or preserve local changes outside this qualification checkout.
- If cleanup fails, retain the structured stderr record and inspect only the invocation-owned workspace and processes. A successful stop plus `codex_temp_clean: true` and `disposable_runtime_removed: true` is the teardown success signal.
