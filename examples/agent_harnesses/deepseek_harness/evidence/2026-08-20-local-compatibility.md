<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DSH local compatibility evidence — 2026-08-20

## Result

The local Tier 0 protocol slice passes. Dynamo normalizes DSH-native root session, immediate parent, and compaction metadata. The lifecycle relay preserves and redacts the request, sends exactly one canonical final per observed session after normal exit and SIGINT, returns 130 after an interrupted drain, and fails closed when the terminal endpoint rejects cleanup.

## Commands and outcomes

| Command | Outcome |
| --- | --- |
| `node --test .agents/skills/dynamo-agent-harness/scripts/test_drive_deepseek_harness.mjs` | 3 passed |
| `cargo test -p dynamo-llm --no-default-features agent_context_from_deepseek_harness_headers_preserves_compaction` | 1 passed |
| `cargo test -p dynamo-llm --no-default-features agent_context_from_headers_derives_agent_context_table` | 1 passed |
| `cargo test -p dynamo-llm --no-default-features session_affinity_prefers_dynamo_header_over_agent_mappings` | 1 passed |
| DSH focused Vitest for adapter and request reconstruction | 115 passed |
| DSH `pnpm run typecheck` | passed |
| Published `@deepseek-ai/dsh@0.1.0-rc.8` headless through the relay and mock Dynamo SSE | passed |
| Built patched DSH source headless through the relay and mock Dynamo SSE | passed |
| DSH Markdown wrap, Agent Note, translation-pair, and generated Cordis catalog checks | passed |
| `kubectl kustomize` plus client-side apply dry-run | Namespace, ConfigMap, and Job passed |
| Dynamo staged-file pre-commit suite | passed |

## Exact contract proved

- Ordinary DSH requests carry `x-deepseek-harness-session-id`; patched child requests additionally carry `x-deepseek-harness-parent-session-id`.
- A DSH auxiliary summary request carries exact `x-deepseek-harness-compact: 1`; other values do not set compaction.
- Dynamo derives input trigger from Chat Completions messages and gives canonical `x-dynamo-session-id` precedence over harness-native mappings.
- The relay redacts Authorization, hashes the stable anonymous user ID, and retains session, parent, compaction, body, response status, and terminal status.
- Terminal delivery uses `x-dynamo-session-id` plus `x-dynamo-session-final: true` and is opt-in because only ThunderAgent consumes it as lifecycle metadata.

## Remaining qualification

No claim is made here about real-model tool quality, a live DSH child run, stock-KV route telemetry, ThunderAgent route telemetry, or post-final program count. Those require the project-owned NScale backend and client job. The local Docker daemon was unavailable, so the two pinned Dockerfiles were reviewed but not built locally. Cluster allocation was inspected read-only; no resources had been created when this report was written.
