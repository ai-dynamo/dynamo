---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning
subtitle: Configure reasoning parsers for models that emit thinking content
---

Some models emit reasoning or thinking content separately from their final response. Dynamo can split that output into `reasoning_content` and normal assistant content by configuring `--dyn-reasoning-parser` on the backend worker.

> [!TIP]
> This page covers parser names for the default Dynamo-native path. For a
> comparison of all preprocessing options (including vLLM/SGLang chat-processor
> swap and tokenizer delegation) and routing
> compatibility, see [Chat Processor Options](chat-processor-options.md).

## Prerequisites

To enable reasoning parsing, launch the backend worker with:

- `--dyn-reasoning-parser`: select the reasoning parser from the supported list below

```bash
# <backend> can be sglang, trtllm, vllm, etc. based on your installation
python -m dynamo.<backend> --help
```

> [!TIP]
> Some models need both a reasoning parser and a tool call parser. For supported tool call parser names, see [Tool Calling](tool-calling.md).

## Supported Reasoning Parsers

The table below lists the currently supported reasoning parsers in Dynamo's registry. The
**Upstream name** column shows where the vLLM or SGLang parser name differs
from Dynamo's -- relevant when using `--dyn-chat-processor vllm` or `sglang`
(see [Chat Processor Options](chat-processor-options.md)). A blank upstream
column means the same name works everywhere. `Dynamo-only` means no upstream
parser exists for this format.

Parsers marked **force-reasoning** emit reasoning content from token one
without requiring an explicit opening tag (`<think>`, etc.). All others
require the opening tag to be present in the model output.

| Parser Name | Models | Upstream name | Force-reasoning | Notes |
|---|---|---|---|---|
| `basic` | Generic CoT models | Dynamo-only | No | Plain `<think>...</think>` |
| `deepseek_r1` | DeepSeek R1, DeepSeek V3.1, DeepSeek V3.2 | | Yes | Pass explicitly for V3.1/V3.2 (no alias) |
| `deepseek_v4` | DeepSeek V4 Pro / Flash | vLLM: `deepseek_v4`; SGLang: `deepseek-v4` | No | `<think>...</think>`. Aliases: `deepseek-v4`, `deepseekv4` |
| `gemma4` | Google Gemma 4 (thinking models) | vLLM: `gemma4` | No | `<\|channel>thought\n...<channel\|>` with `thought\n` role label stripped. Aliases: `gemma-4` |
| `glm45` | GLM-4.5, GLM-4.7 | Dynamo-only | No | Alias for `nemotron_deci`. `<think>...</think>` |
| `gpt_oss` | gpt-oss-20b / -120b | Dynamo-only | No | Harmony channel reasoning format |
| `granite` | Granite 3.x | | No | `Here's my thought process:` / `Here's my response:` |
| `kimi` | Kimi K2 Instruct / Thinking | Dynamo-only | No | `◁think▷...◁/think▷` |
| `kimi_k25` | Kimi K2.5 | Dynamo-only | Yes | `<think>...</think>` with force-reasoning |
| `minimax_append_think` | MiniMax M2 / M2.1 | Dynamo-only | No | Implicit opening `<think>` prepended |
| `mistral` | Magistral | | Yes | `[THINK]...[/THINK]` |
| `nemotron3` | Nemotron-3 / Mini | Dynamo-only | Yes | Alias for `deepseek_r1` |
| `nemotron_deci` | Nemotron-Super / -Ultra / -Deci, Llama-Nemotron | Dynamo-only | No | `<think>...</think>` |
| `nemotron_nano` | Nemotron-Nano | Dynamo-only | Yes | Alias for `deepseek_r1` |
| `qwen3` | QwQ-32B, Qwen3-Think, Qwen3-Coder, Qwen3.5 | | No | `<think>...</think>` |
| `step3` | Step-3 / Step-3-Reasoning | Dynamo-only | Yes | `<think>...</think>` |

## Common Parser Pairings

Some models need both parsers configured together. Common pairings include:

- `openai/gpt-oss-*`: `--dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss`
- `deepseek-ai/DeepSeek-V4-*`: `--dyn-tool-call-parser deepseek_v4 --dyn-reasoning-parser deepseek_v4`
- `zai-org/GLM-4.7`: `--dyn-tool-call-parser glm47 --dyn-reasoning-parser glm45`
- `moonshotai/Kimi-K2.5*`: `--dyn-tool-call-parser kimi_k2 --dyn-reasoning-parser kimi_k25`
- `google/gemma-4-*` thinking models: `--dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4 --custom-jinja-template examples/chat_templates/gemma4_tool.jinja`
- `Qwen/Qwen3.5*`: `--dyn-tool-call-parser qwen3_coder --dyn-reasoning-parser qwen3`
- MiniMax M2.1 style outputs: `--dyn-tool-call-parser minimax_m2 --dyn-reasoning-parser minimax_append_think`

## Tool Calling Interplay

Reasoning parsing happens before tool call parsing. If a model emits both reasoning content and tool calls, configure both parsers so Dynamo can first separate reasoning text and then parse tool calls from the remaining assistant output.
