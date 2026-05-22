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
parser lives in Dynamo's own registry or in an upstream engine frontend
(`vllm serve`, `sglang serve`, or `trtllm-serve`).

## Choose a parsing path

| Path | When to use | Page |
|------|-------------|------|
| **Dynamo** | Dynamo ships a framework-agnostic Rust parser for the model's tool-call format. Default path. | [Tool Call Parsing (Dynamo)](dynamo.md) |
| **Engine Fallback** | Use the framework's parser implementation (vLLM or SGLang today; TRTLLM in progress) for pre/post processing, including tool call and reasoning parsing - ensure consistency with framework behavior. | [Tool Call Parsing (Engine Fallback)](engine-fallback.md) |

Start with the Dynamo path. Fall back to the engine path only when Dynamo's
registry does not list a parser for your model.

## Why Dynamo implements tool-call and reasoning parsers

In `vllm serve`, `sglang serve`, and `trtllm-serve`, tool-call parsing and
reasoning parsing happens in the engine's frontend server, with subtle
behavioral differences across each. Since Dynamo orchestrates efficient
routing and tokenization, it connects directly to each engine and circumvents
each engine's frontend server to avoid redundant work.

Dynamo therefore implements tool-call parsing and reasoning parsing in its frontend as a
framework-agnostic Rust layer. This gives Dynamo one tested OpenAI-compatible
contract across vLLM, SGLang, TRTLLM, and other workers, while keeping the serving
hot path highly concurrent and scalable, avoiding Python GIL bottlenecks.

## Troubleshooting

If a tool call comes back wrong, add `logprobs: true` to a single repro
request and share the response. See
[Troubleshooting Tool Calls](troubleshooting.md) for what to capture and
include when reporting an issue.

## See Also

- [Troubleshooting Tool Calls](troubleshooting.md) -- capture raw model
  output with `logprobs` so tool-call issues can be localized.
- [Reasoning](../reasoning/README.md) -- separate `reasoning_content` from
  assistant content for chain-of-thought models. Several models need both a
  tool-call parser and a reasoning parser configured together.
- [Frontend Configuration Reference](../components/frontend/configuration.md) --
  full CLI flag reference.
