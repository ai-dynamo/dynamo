---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning
subtitle: Configure reasoning parsers for models that emit thinking content
---

Some models emit reasoning or thinking content separately from their final response. Dynamo can split that output into `reasoning_content` and normal assistant content by configuring `--dyn-reasoning-parser` on the backend worker.

## Prerequisites

To enable reasoning parsing, launch the backend worker with:

- `--dyn-reasoning-parser`: select the reasoning parser from the supported list below

```bash
# <backend> can be sglang, trtllm, vllm, etc. based on your installation
python -m dynamo.<backend> --help
```

<Tip>
Some models need both a reasoning parser and a tool call parser. For supported tool call parser names, see [Tool Calling](tool-calling.md).
</Tip>

## Supported Reasoning Parsers

The reasoning parser names currently supported in the codebase are:

| Parser Name | Typical Models / Format |
|-------------|-------------------------|
| `basic` | Generic `<think>...</think>` reasoning blocks |
| `deepseek_r1` | Models that should treat output as reasoning until `</think>` is seen, such as `deepseek-ai/DeepSeek-R1` style responses |
| `glm45` | `zai-org/GLM-4.5` and GLM-5 style `<think>...</think>` reasoning blocks |
| `gpt_oss` | `openai/gpt-oss-*` |
| `granite` | Granite models that emit `Here's my thought process:` / `Here's my response:` markers |
| `kimi` | Kimi models that emit `◁think▷...◁/think▷` |
| `kimi_k25` | `moonshotai/Kimi-K2.5*` models that require force-reasoning handling for `<think>...</think>` |
| `minimax_append_think` | MiniMax models that begin reasoning immediately and effectively need an implicit opening `<think>` tag prepended |
| `mistral` | Mistral reasoning models that emit `[THINK]...[/THINK]` |
| `nemotron_deci` | Nemotron models that emit standard `<think>...</think>` reasoning blocks |
| `nemotron_nano` | Nemotron Nano reasoning output that ends with `</think>` without requiring a visible opening tag |
| `qwen3` | `Qwen/Qwen3-*` style `<think>...</think>` responses |
| `step3` | Step-style models that should treat content as reasoning until `</think>` is seen |

## Common Parser Pairings

Some models need both parsers configured together. Common pairings include:

- `openai/gpt-oss-*`: `--dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss`
- `zai-org/GLM-4.7`: `--dyn-tool-call-parser glm47 --dyn-reasoning-parser glm45`
- `moonshotai/Kimi-K2.5*`: `--dyn-tool-call-parser kimi_k2 --dyn-reasoning-parser kimi_k25`
- MiniMax M2.1 style outputs: `--dyn-tool-call-parser minimax_m2 --dyn-reasoning-parser minimax_append_think`

## Tool Calling Interplay

Reasoning parsing happens before tool call parsing. If a model emits both reasoning content and tool calls, configure both parsers so Dynamo can first separate reasoning text and then parse tool calls from the remaining assistant output.
