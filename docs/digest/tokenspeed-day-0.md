---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Dynamo Day 0 support for TokenSpeed"
subtitle: "A short launch note for running TokenSpeed with Dynamo"
description: "Dynamo adds day-0 TokenSpeed support with the Dynamo frontend for Kimi K2.5."
keywords: TokenSpeed, Dynamo, LightSeek, Kimi K2.5, disaggregated serving, KV cache routing, LLM inference
last-updated: May 6, 2026
---

[TokenSpeed](https://lightseek.org/blog/lightseek-tokenspeed.html) ([GitHub](https://github.com/lightseekorg/tokenspeed)) launched today as LightSeek's new inference engine for agentic workloads. The initial repo is a preview, with more model coverage and runtime features landing over the next few weeks.

Two pieces are worth calling out. First, TokenSpeed includes new MLA kernel work for long-context Kimi-style workloads on Blackwell. Second, TokenSpeed has a native C++ scheduler in `tokenspeed-scheduler/` that models request flow and cache operations as explicit state machines, while Python remains the runtime and integration layer.

Dynamo now has day-0 support for running TokenSpeed as a Dynamo backend through `python -m dynamo.tokenspeed`. The Dynamo frontend remains the user-facing OpenAI-compatible API entrypoint and handles request routing, streaming responses, and cancellation.

See the [Kimi K2.5 TokenSpeed recipe](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k2.5/tokenspeed/agg/nvidia) for the current Dynamo launch recipe.

Things are moving quickly. Upstream TokenSpeed calls out ongoing work on model coverage, P/D, EPLB, KV store, Mamba cache, VLM, metrics, Hopper optimization, and related runtime features.
