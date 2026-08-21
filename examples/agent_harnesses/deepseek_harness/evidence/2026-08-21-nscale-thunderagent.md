<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek Harness NScale ThunderAgent evidence — 2026-08-21

## Result

The published `@deepseek-ai/dsh@0.1.0-rc.8` package completed a live one-shot turn through the public ThunderAgent model route and drained its observed session through the model-hidden terminal path. The child exited 0, both streamed model responses returned HTTP 200, the terminal response returned HTTP 200, and the relay reported `final_failed: false`.

## Frozen inputs

- Dynamo runtime image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:effd250754b8a70517c27eab8f18463b395a7b2a8e868fd919226c3180636939`
- Model: `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, 32,768-token context
- DSH package: `@deepseek-ai/dsh@0.1.0-rc.8`
- DSH session: `session-aa5d0507-f889-45a5-bdaf-be1775584392`
- Raw redacted trace: [`2026-08-21-nscale-thunderagent.jsonl`](2026-08-21-nscale-thunderagent.jsonl)

## Command contract and outcome

The run enabled `--canonicalize-dynamo-headers` for the pinned Dynamo 1.3 compatibility bridge and `--session-final` only after the endpoint was switched from stock KV to ThunderAgent. Native DSH identity remained present while the bridge copied it to the canonical Dynamo header used by that older runtime.

The model returned `DSH_THUNDERAGENT_OK.` with a trailing period. That is a successful live transport response but a one-character miss against the exact-token oracle, so this evidence does not claim perfect instruction following. It does prove DSH child startup, streamed Chat Completions transport, stable session identity, normal process exit, and terminal delivery.

The router independently logged `Released program session-aa5d0507-f889-45a5-bdaf-be1775584392 (0 remaining)`. The pinned runtime predates the newer labeled `thunderagent.route path=...` messages, so the lifecycle acceptance pair is the relay's HTTP 200 `session_final` record plus the same-ID router release with zero programs remaining.

## Scope boundary

This is the published-package root-session path. Immediate child lineage remains qualified by the isolated DSH source patch and local focused tests; the live tiny-model prompt did not create a child session. Tool quality and nontrivial coding-task correctness require a stronger model and are not claimed.
