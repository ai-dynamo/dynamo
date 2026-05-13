---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Calling
subtitle: Parse tool calls from model output and surface them as OpenAI-compatible tool_calls
---

Dynamo can connect models to external tools and services by parsing tool-call
syntax out of raw model output and surfacing it as OpenAI-compatible
`tool_calls` on the response. Tool calling is controlled by the `tool_choice`
and `tools` request parameters on the chat completions API.

There are two ways to parse tool calls in Dynamo, depending on whether the
parser lives in Dynamo's own registry or in the upstream engine (vLLM, SGLang).

## Choose a parsing path

| Path | When to use | Page |
|------|-------------|------|
| **Dynamo** | Dynamo ships a Rust parser for the model's tool-call format. Lowest latency, the default path. | [Tool Call Parsing (Dynamo)](dynamo.md) |
| **Engine Fallback** | Use the framework's implementation (vLLM or SGLang) for pre/post processing, including tool call and reasoning parsing - ensure consistency with framework behavior. | [Tool Call Parsing (Engine Fallback)](engine-fallback.md) |

Start with the Dynamo path. Fall back to the engine path only when Dynamo's
registry does not list a parser for your model.

## See Also

- [Reasoning](../reasoning/README.md) -- separate `reasoning_content` from
  assistant content for chain-of-thought models. Several models need both a
  tool-call parser and a reasoning parser configured together.
- [Frontend Configuration Reference](../components/frontend/configuration.md) --
  full CLI flag reference.
