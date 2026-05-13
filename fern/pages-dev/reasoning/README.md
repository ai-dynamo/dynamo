---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning
subtitle: Separate reasoning content from assistant output for chain-of-thought models
---

Some models emit reasoning or thinking content separately from their final
response. Dynamo can split that output into `reasoning_content` and normal
assistant content by configuring a reasoning parser. As with tool calling,
there are two ways to do this to ensure wide coverage and day0 model support.

## Choose a parsing path

| Path | When to use | Page |
|------|-------------|------|
| **Dynamo** | Dynamo ships a Rust parser for the model's reasoning format. Lowest latency, the default path. | [Reasoning Parsing (Dynamo)](dynamo.md) |
| **Engine Fallback** | Use the framework's implementation (vLLM or SGLang) for pre/post processing, including tool call and reasoning parsing - ensure consistency with framework behavior. | [Reasoning Parsing (Engine Fallback)](engine-fallback.md) |

Start with the Dynamo path. Fall back to the engine path only when Dynamo's
registry does not list a parser for your model.

## See Also

- [Tool Calling](../tool-calling/README.md) -- parse tool calls out of model
  output. Several models need both a reasoning parser and a tool-call parser
  configured together.
- [Frontend Configuration Reference](../components/frontend/configuration.md) --
  full CLI flag reference.
