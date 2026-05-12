---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Calling
subtitle: Connect Dynamo to external tools and services using function calling
---

You can connect Dynamo to external tools and services using function calling (also known as tool calling). By providing a list of available functions, Dynamo can choose
to output function arguments for the relevant function(s) which you can execute to augment the prompt with relevant external information.

Tool calling (AKA function calling) is controlled using the `tool_choice` and `tools` request parameters.

> [!TIP]
> This page covers parser names for the default Dynamo-native path. For a
> comparison of all preprocessing options (including vLLM/SGLang chat-processor
> swap and tokenizer delegation) and routing
> compatibility, see [Chat Processor Options](chat-processor-options.md).

## Prerequisites

To enable this feature, you should set the following flag while launching the backend worker

- `--dyn-tool-call-parser`: select the tool call parser from the supported list below

```bash
# <backend> can be sglang, trtllm, vllm, etc. based on your installation
python -m dynamo.<backend> --help
```

> [!NOTE]
> If no tool call parser is provided by the user, Dynamo will try to use default tool call parsing based on &lt;TOOLCALL&gt; and &lt;|python_tag|&gt; tool tags.

> [!TIP]
> If your model's default chat template doesn't support tool calling, but the model itself does, you can specify a custom chat template per worker
> with `python -m dynamo.<backend> --custom-jinja-template </path/to/template.jinja>`.

> [!TIP]
> If your model also emits reasoning content that should be separated from normal output, see [Reasoning](reasoning.md) for the supported `--dyn-reasoning-parser` values.

## Supported Tool Call Parsers

The table below lists the currently supported tool call parsers in Dynamo's registry. The
**Upstream name** column shows where the vLLM or SGLang parser name differs
from Dynamo's -- relevant when using `--dyn-chat-processor vllm` or `sglang`
(see [Chat Processor Options](chat-processor-options.md)). A blank upstream
column means the same name works everywhere. `Dynamo-only` means no upstream
parser exists for this format.

| Parser Name | Models | Upstream name | Notes |
|---|---|---|---|
| `deepseek_v3` | DeepSeek V3, DeepSeek R1-0528+ | SGLang: `deepseekv3` | Special Unicode markers |
| `deepseek_v3_1` | DeepSeek V3.1 | Dynamo-only | JSON separators |
| `deepseek_v3_2` | DeepSeek V3.2+ | Dynamo-only | DSML tags (`<｜DSML｜function_calls>...`) |
| `deepseek_v4` | DeepSeek V4 Pro / Flash | vLLM: `deepseek_v4`; SGLang: `deepseekv4` | DSML tags (`<｜DSML｜tool_calls>...`). Aliases: `deepseek-v4`, `deepseekv4` |
| `default` | *(fallback)* | Dynamo-only | Empty JSON config (no start/end tokens). Prefer a model-specific parser for production use. |
| `gemma4` | Google Gemma 4 (thinking models) | vLLM: `gemma4` | Custom non-JSON grammar with `<\|"\|>` string delimiters and `<\|tool_call>...<tool_call\|>` markers. Aliases: `gemma-4`. Pair with `--dyn-reasoning-parser gemma4` and `--custom-jinja-template examples/chat_templates/gemma4_tool.jinja` |
| `glm47` | GLM-4.5, GLM-4.7 | Dynamo-only | XML `<arg_key>/<arg_value>` |
| `harmony` | gpt-oss-20b / -120b | Dynamo-only | Harmony channel format |
| `hermes` | Qwen2.5-\*, QwQ-32B, Qwen3-Instruct, Qwen3-Think, NousHermes-2/3 | vLLM: `qwen2_5`; SGLang: `qwen25` (for Qwen models) | `<tool_call>` JSON |
| `jamba` | Jamba 1.5 / 1.6 / 1.7 | Dynamo-only | `<tool_calls>` JSON |
| `kimi_k2` | Kimi K2 Instruct/Thinking, Kimi K2.5 | | Pair with `--dyn-reasoning-parser kimi` or `kimi_k25` |
| `llama3_json` | Llama 3 / 3.1 / 3.2 / 3.3 Instruct | | `<\|python_tag\|>` tool syntax |
| `minimax_m2` | MiniMax M2 / M2.1 | vLLM: `minimax` | XML `<minimax:tool_call>` |
| `mistral` | Mistral / Mixtral / Mistral-Nemo, Magistral | | `[TOOL_CALLS]...[/TOOL_CALLS]` |
| `nemotron_deci` | Nemotron-Super / -Ultra / -Deci, Llama-Nemotron-Ultra / -Super | Dynamo-only | `<TOOLCALL>` JSON |
| `nemotron_nano` | Nemotron-Nano | Dynamo-only | Alias for `qwen3_coder` |
| `phi4` | Phi-4, Phi-4-mini, Phi-4-mini-reasoning | vLLM: `phi4_mini_json` | `functools[...]` JSON |
| `pythonic` | Llama 4 (Scout / Maverick) | | Python-list tool syntax |
| `qwen3_coder` | Qwen3-Coder, Qwen3.5 | | XML `<tool_call><function=...>` |

> [!TIP]
> For Kimi K2.5 thinking models, pair `--dyn-tool-call-parser kimi_k2` with
> `--dyn-reasoning-parser kimi_k25` from [Reasoning](reasoning.md) so that both `<think>` blocks and tool calls
> are parsed correctly from the same response.

## Examples

### Launch Dynamo Frontend and Backend

```bash
# launch backend worker
python -m dynamo.vllm --model openai/gpt-oss-20b --dyn-tool-call-parser harmony

# launch frontend worker
python -m dynamo.frontend
```

### Tool Calling Request Examples

- Example 1
```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8081/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."
tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco in Celsius?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=10000
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
```

- Example 2
```python

# Use tools defined in example 1

time_tool = {
    "type": "function",
    "function": {
        "name": "get_current_time_nyc",
        "description": "Get the current time in NYC.",
        "parameters": {}
    }
}


tools.append(time_tool)

messages = [
    {"role": "user", "content": "What's the current time in New York?"}
]


response = client.chat.completions.create(
    model="openai/gpt-oss-20b", #client.models.list().data[1].id,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=100,
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```

- Example 3


```python

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_tourist_attractions",
            "description": "Get a list of top tourist attractions for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to find attractions for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

def get_messages():
    return [
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. what are some top tourist attractions in Tokyo? "
            ),
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=100,
)
print(f"{response}")
tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
```
