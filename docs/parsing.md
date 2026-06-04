---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Call and Reasoning Parsing
subtitle: Parse tool calls and reasoning out of model output into OpenAI-compatible tool_calls and reasoning_content
---

Dynamo parses tool-call and reasoning markup out of raw model output and surfaces it as OpenAI-compatible `tool_calls` and `reasoning_content` on the response. Tool calling is controlled by the `tool_choice` and `tools` request parameters; reasoning parsing is enabled per-model with a reasoning parser.

There are two ways to parse, depending on whether the parser lives in Dynamo's own registry or in an upstream engine frontend (`vllm serve`, `sglang serve`, or `trtllm-serve`).

## Choose a parsing path

| Path | When to use | Pages |
|------|-------------|-------|
| **Dynamo** | Dynamo ships a framework-agnostic Rust parser for the model's tool-call or reasoning format. Default path. | [Tool Call Parsing (Dynamo)](tool-calling/README.md), [Reasoning Parsing (Dynamo)](reasoning/README.md) |
| **Engine Fallback** | Use the framework's own parser (vLLM or SGLang today; TRT-LLM in progress) when Dynamo doesn't ship one for your model. | [Parser Engine Fallback](engine-fallback.md) |

Start with the Dynamo path. Fall back to the engine path only when Dynamo's registry doesn't list a parser for your model. For exactly which flags combine and which combinations don't make sense, see [Parser Configuration](parser-configuration.md).

## Why Dynamo parses in the frontend

In `vllm serve`, `sglang serve`, and `trtllm-serve`, tool-call and reasoning parsing happen in each engine's own frontend, with subtle behavioral differences across them. For performance, Dynamo orchestrates routing and tokenization and passes tokens directly to each engine, bypassing the engine's OpenAI server to avoid duplicate work per request. So Dynamo implements parsing in its frontend as a framework-agnostic Rust layer — one tested OpenAI-compatible contract across vLLM, SGLang, and TRT-LLM, on a hot path that stays concurrent without a Python GIL bottleneck. The `vllm`/`sglang` chat processors (engine fallback) opt back into the engine's own parser when Dynamo doesn't ship one for your model.

## See Also

- [Parser Configuration](parser-configuration.md) -- which flags combine, and which combinations don't make sense
- [Tool Call Parsing (Dynamo)](tool-calling/README.md) / [Reasoning Parsing (Dynamo)](reasoning/README.md) -- Dynamo-native parser names
- [Parser Engine Fallback](engine-fallback.md) -- upstream vLLM / SGLang parsers
- [Troubleshooting Tool Calls](tool-calling/troubleshooting.md) -- capture `logprobs` so issues can be localized
- [Frontend Configuration Reference](components/frontend/configuration.md) -- full CLI flag reference
