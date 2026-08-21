<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSH local compatibility evidence — 2026-08-20

## Result

The local Tier 0 protocol slice passes, and the stock NScale arm reaches the pinned real model through Dynamo. Dynamo normalizes DSH-native root-session and compaction metadata only when DSH identity is selected. The lifecycle relay passes a minimal child environment, ignores ambient DeepSeek credentials, preserves and redacts each request, uses exclusive capture creation, fails when lifecycle mode observes no sessions, sends exactly one canonical final per observed session, and terminates the complete tracked Corepack/pnpm/DSH process tree. Interrupted drains retain exit code `130` for SIGINT and `143` for SIGTERM.

## Commands and outcomes

| Command | Outcome |
| --- | --- |
| `DSH_PACKAGE_SMOKE=1 node --test .agents/skills/dynamo-agent-harness/scripts/test_drive_deepseek_harness.mjs` | 10 passed |
| `cargo test -p dynamo-llm --no-default-features agent_context_from_deepseek_harness_headers_preserves_compaction` | 1 passed |
| `cargo test -p dynamo-llm --no-default-features deepseek_compaction_requires_selected_deepseek_identity` | 1 passed |
| `cargo test -p dynamo-llm --no-default-features agent_context_from_headers_derives_agent_context_table` | 1 passed |
| `cargo test -p dynamo-llm --no-default-features session_affinity_prefers_dynamo_header_over_agent_mappings` | 1 passed |
| Published `@deepseek-ai/dsh@0.1.0-rc.8` headless through the hardened relay and mock Dynamo SSE | passed |
| Real `corepack pnpm dlx` wrapper with an uncooperative detached descendant, followed by SIGTERM | tracked tree received SIGTERM, detached descendant required bounded SIGKILL, wrapper returned `143`, and terminal request drained |
| `kubectl kustomize` plus client-side apply dry-run | Namespace, ConfigMap, and Job passed |
| `corepack pnpm@11.7.0 --dir examples/agent_harnesses/deepseek_harness/client install --prod --frozen-lockfile --ignore-scripts --lockfile-only` | passed supply-chain policy and frozen-lock checks |
| `cargo +stable fmt --all -- --check` | passed; the branch-pinned custom toolchain has no applicable rustfmt component on this host |
| `uvx pre-commit run --files ...` across every changed path | passed |

## Stock NScale evidence

The relay ran the published DSH package against the project-owned stock Dynamo graph at `http://127.0.0.1:18000`, serving `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` from the pinned Dynamo runtime image. The run used `--canonicalize-dynamo-headers` because the runtime predates this branch's native DSH mapping. The redacted raw trace is [`2026-08-20-nscale-stock.jsonl`](2026-08-20-nscale-stock.jsonl).

The trace contains two streamed `POST /v1/chat/completions` requests under native session `session-4e5f78b5-81ab-4e55-9188-c7d1fb6c68fc`; both responses returned HTTP 200, Authorization is redacted, the model is exact, and the DSH child exited 0 without a signal. The prompt requested the exact token `DSH_STOCK_OK`, but the 0.6B model emitted a DSH skill directive instead. This proves live harness, relay, Dynamo, and model transport, but not instruction-following or tool correctness for the tiny qualification model.

The first corrected live attempt also exposed a relay defect: URL assignment through `URL.pathname` retained an extra slash for a root endpoint and produced `//v1`, which DSH interpreted as a host named `v1`. The normalized URL path now strips trailing slashes before appending `/v1`; deterministic tests cover both root and already-versioned inputs. A later transient failure was traced to an expired local port-forward, not the harness, and was replaced by the successful raw trace above.

## Exact contract proved

- Ordinary DSH requests carry `x-deepseek-harness-session-id`; the published-package path makes no immediate-parent-lineage claim.
- A request selected through DSH identity carries compaction metadata only for exact `x-deepseek-harness-compact: 1`; mixed canonical, Codex, and Claude Code identities ignore the DSH compaction header.
- Dynamo derives input trigger from Chat Completions messages and gives canonical `x-dynamo-session-id` precedence over harness-native mappings.
- The relay selects `DYNAMO_API_KEY` rather than ambient `DEEPSEEK_API_KEY`, strips an injected unrelated credential from the child environment, redacts Authorization, hashes the stable anonymous user ID, and retains session, compaction, body, response status, and terminal status.
- The opt-in Dynamo 1.3 bridge preserves native headers while adding canonical session identity. It does not claim older-server compaction normalization.
- Terminal delivery uses `x-dynamo-session-id` plus `x-dynamo-session-final: true` and is opt-in because only ThunderAgent consumes it as lifecycle metadata.

## Remaining qualification

No claim is made here about real-model tool quality, stock-KV route telemetry, ThunderAgent route telemetry, or post-final program count. The stock NScale arm proves a live DSH child and model response only. ThunderAgent lifecycle evidence remains a separate sequential arm. The local Docker daemon was unavailable, so the pinned package-only Dockerfile was reviewed but not built locally.
