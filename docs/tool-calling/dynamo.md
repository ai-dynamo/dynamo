---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Call Parsing (Dynamo)
subtitle: Connect Dynamo to external tools and services using Dynamo's built-in tool call parsers
---

You can connect Dynamo to external tools and services using tool calling. By
providing a list of available functions, Dynamo can choose to output function
arguments for the relevant function(s) which you can execute to augment the
prompt with relevant external information.

Tool calling is controlled using the `tool_choice` and `tools` request
parameters.

This page covers parser names for the default Dynamo-native path. If Dynamo
does not list a parser for your model, see
[Tool Call Parsing (Engine Fallback)](engine-fallback.md).

## Prerequisites

To enable this feature, you should set the following flag while launching the backend worker

- `--dyn-tool-call-parser`: select the tool call parser from the supported list below

```bash
# <backend> can be sglang, trtllm, vllm, etc. based on your installation
python -m dynamo.<backend> --help
```

> [!TIP]
> If your model's default chat template doesn't support tool calling, but the model itself does, you can specify a custom chat template per worker
> with `python -m dynamo.<backend> --custom-jinja-template </path/to/template.jinja>`.

> [!TIP]
> If your model also emits reasoning content that should be separated from normal output, see [Reasoning Parsing (Dynamo)](../reasoning/dynamo.md) for the supported `--dyn-reasoning-parser` values.

## Supported Tool Call Parsers

The table below lists the currently supported tool call parsers in Dynamo's registry. The
**Upstream name** column shows where the vLLM or SGLang parser name differs
from Dynamo's -- relevant when using `--dyn-chat-processor vllm` or `sglang`
(see [Tool Call Parsing (Engine Fallback)](engine-fallback.md)). A blank upstream
column means the same name works everywhere. `Dynamo-only` means no upstream
parser exists for this format.

| Parser Name | Models | Upstream name | Notes |
|---|---|---|---|
| `kimi_k2` | Kimi K2 Instruct/Thinking, Kimi K2.5 | | Pair with `--dyn-reasoning-parser kimi` or `kimi_k25` |
| `qwen3_coder` | Qwen3.5, Qwen3-Coder | | XML `<tool_call><function=...>` |
| `deepseek_v4` | DeepSeek V4 Pro / Flash | vLLM: `deepseek_v4`; SGLang: `deepseekv4` | DSML tags (`<｜DSML｜tool_calls>...`). Aliases: `deepseek-v4`, `deepseekv4` |
| `deepseek_v3` | DeepSeek V3, DeepSeek R1-0528+ | SGLang: `deepseekv3` | Special Unicode markers |
| `deepseek_v3_1` | DeepSeek V3.1 | Dynamo-only | JSON separators |
| `deepseek_v3_2` | DeepSeek V3.2+ | Dynamo-only | DSML tags (`<｜DSML｜function_calls>...`) |
| `default` | *(fallback)* | Dynamo-only | Empty JSON config (no start/end tokens). Prefer a model-specific parser for production use. |
| `gemma4` | Google Gemma 4 (thinking models) | vLLM: `gemma4` | Custom non-JSON grammar with `<\|"\|>` string delimiters and `<\|tool_call>...<tool_call\|>` markers. Aliases: `gemma-4`. Pair with `--dyn-reasoning-parser gemma4` and `--custom-jinja-template examples/chat_templates/gemma4_tool.jinja` |
| `glm47` | GLM-4.5, GLM-4.7 | Dynamo-only | XML `<arg_key>/<arg_value>` |
| `harmony` | gpt-oss-20b / -120b | Dynamo-only | Harmony channel format |
| `hermes` | Qwen2.5-\*, QwQ-32B, Qwen3-Instruct, Qwen3-Think, NousHermes-2/3 | vLLM: `qwen2_5`; SGLang: `qwen25` (for Qwen models) | `<tool_call>` JSON |
| `jamba` | Jamba 1.5 / 1.6 / 1.7 | Dynamo-only | `<tool_calls>` JSON |
| `llama3_json` | Llama 3 / 3.1 / 3.2 / 3.3 Instruct | | `<\|python_tag\|>` tool syntax |
| `minimax_m2` | MiniMax M2 / M2.1 | vLLM: `minimax` | XML `<minimax:tool_call>` |
| `mistral` | Mistral / Mixtral / Mistral-Nemo, Magistral | | `[TOOL_CALLS]...[/TOOL_CALLS]` |
| `nemotron_deci` | Nemotron-Super / -Ultra / -Deci, Llama-Nemotron-Ultra / -Super | Dynamo-only | `<TOOLCALL>` JSON |
| `nemotron_nano` | Nemotron-Nano | Dynamo-only | Alias for `qwen3_coder` |
| `phi4` | Phi-4, Phi-4-mini, Phi-4-mini-reasoning | vLLM: `phi4_mini_json` | `functools[...]` JSON |
| `pythonic` | Llama 4 (Scout / Maverick) | | Python-list tool syntax |

> [!TIP]
> For Kimi K2.5 thinking models, pair `--dyn-tool-call-parser kimi_k2` with
> `--dyn-reasoning-parser kimi_k25` from [Reasoning Parsing (Dynamo)](../reasoning/dynamo.md) so that both `<think>` blocks and tool calls
> are parsed correctly from the same response.

## Examples

### Launch Dynamo Frontend and Backend

```bash
# launch backend worker (or dynamo.vllm)
python -m dynamo.sglang --model Qwen/Qwen3.5-4B --dyn-tool-call-parser qwen3_coder --dyn-reasoning-parser qwen3

# launch frontend worker
python -m dynamo.frontend
```

### Tool Calling Request Example

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco and New York?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

Dynamo parses the tool calls out of the model output and surfaces them as
OpenAI-compatible `tool_calls` entries on the response:

```json
{
  "id": "chatcmpl-b415caad-9be0-4d9e-ac6d-9d23bfe57703",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "reasoning_content": "The user is asking about the weather in two cities: San Francisco and New York. I need to call the get_weather function for each city. I'll make two separate function calls to get the weather information for both locations.\n",
        "tool_calls": [
          {
            "id": "call-56223a95-3d14-4433-a94e-011f106c0e40",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"San Francisco\"}"
            }
          },
          {
            "id": "call-d5b5772b-6b0c-4120-ad01-623278a937fe",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"New York\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls",
      "logprobs": null
    }
  ],
  "created": 1778653281,
  "model": "Qwen/Qwen3.5-4B",
  ...
}
```
