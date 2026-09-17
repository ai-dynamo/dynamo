<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Networked frontend-to-vLLM smoke

This experiment exercises lifecycle headers through Dynamo's public HTTP frontend, post-selection KV hint policy, distributed worker endpoint, Python vLLM handler, and vLLM request-completion action.

The experiment-only activation hook is in `lib/bindings/python/rust/llm/entrypoint.rs` and is disabled unless `DYN_EXPERIMENTAL_SESSION_KV_HINT_POLICY` is set. The linked policy implementation is under `../policy/`.

The exact configuration, measured results, and limitations are recorded in `dynamo-workflows/dynamo/routing/agentic-kv-management/experiments/continuum-kv-hints/report.md`.
