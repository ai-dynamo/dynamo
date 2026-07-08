---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 工具调用解析（Dynamo）
subtitle: 使用 Dynamo 内置的工具调用解析器，将模型连接到外部工具和服务
---

你可以通过工具调用把 Dynamo 连接到外部工具和服务。通过提供一组可用函数，Dynamo 可以为相关函数输出函数参数，你执行这些函数后，即可用外部信息来增强提示。

工具调用由 `tool_choice` 和 `tools` 请求参数控制。

本页介绍默认的 Dynamo 原生路径的解析器名称。如果 Dynamo 未列出适用于你的模型的解析器，请参阅
[Parser Engine Fallback](/dynamo/dev/user-guides/parsing/parser-engine-fallback)。关于 `--dyn-tool-call-parser` 如何与
`--dyn-chat-processor` 和 `--dyn-reasoning-parser` 组合（以及哪些组合是无效的），请参阅
[Parser Configuration](/dynamo/dev/user-guides/parsing/parser-configuration)。

## 前置条件

启动后端 worker 时设置以下 flag 即可启用该功能：

- `--dyn-tool-call-parser`：从下方支持列表中选择工具调用解析器

```bash
# <backend> 可以是 sglang、trtllm、vllm 等，取决于你的安装
python -m dynamo.<backend> --help
```

<Tip>
如果你的模型默认的 chat template 不支持工具调用，但模型本身支持，你可以为每个 worker 指定自定义 chat template：
`python -m dynamo.<backend> --custom-jinja-template </path/to/template.jinja>`。
</Tip>

<Tip>
如果你的模型还会输出需要与正常内容分离的推理内容，请参阅 [Reasoning Parsing (Dynamo)](/dynamo/dev/user-guides/parsing/reasoning-parsing-dynamo) 了解支持的 `--dyn-reasoning-parser` 取值。
</Tip>

## 支持的工具调用解析器

下表列出 Dynamo 注册表中当前支持的工具调用解析器。**Upstream name** 列标出 vLLM 或 SGLang 的解析器名称与 Dynamo 不同之处——在使用 `--dyn-chat-processor vllm` 或 `sglang` 时（参阅 [Parser Engine Fallback](/dynamo/dev/user-guides/parsing/parser-engine-fallback)）尤为相关。upstream 列为空表示同名在各处通用。`Dynamo-only` 表示该格式没有对应的上游解析器。

| 解析器名称 | 模型 | Upstream name | 说明 |
|---|---|---|---|
| `kimi_k2` | Kimi K2 Instruct/Thinking, Kimi K2.5 | | 与 `--dyn-reasoning-parser kimi` 或 `kimi_k25` 配对 |
| `minimax_m2` | MiniMax M2 / M2.1 | vLLM: `minimax` | XML `<minimax:tool_call>` |
| `deepseek_v4` | DeepSeek V4 Pro / Flash | vLLM: `deepseek_v4`; SGLang: `deepseekv4` | DSML 标签（`<｜DSML｜tool_calls>...`）。别名：`deepseek-v4`、`deepseekv4` |
| `deepseek_v3` | DeepSeek V3, DeepSeek R1-0528+ | SGLang: `deepseekv3` | 特殊 Unicode 标记 |
| `deepseek_v3_1` | DeepSeek V3.1 | Dynamo-only | JSON 分隔符 |
| `deepseek_v3_2` | DeepSeek V3.2+ | Dynamo-only | DSML 标签（`<｜DSML｜function_calls>...`） |
| `qwen3_coder` | Qwen3.5, Qwen3-Coder | | XML `<tool_call><function=...>` |
| `glm47` | GLM-4.5, GLM-4.7 | Dynamo-only | XML `<arg_key>/<arg_value>` |
| `nemotron_deci` | Nemotron-Super / -Ultra / -Deci, Llama-Nemotron-Ultra / -Super | Dynamo-only | `<TOOLCALL>` JSON |
| `nemotron_nano` | Nemotron-Nano | Dynamo-only | `qwen3_coder` 的别名 |
| `gemma4` | Google Gemma 4（thinking 模型） | vLLM: `gemma4` | 自定义非 JSON 文法，使用 `<\|"\|>` 字符串分隔符和 `<\|tool_call>...<tool_call\|>` 标记。别名：`gemma-4`。与 `--dyn-reasoning-parser gemma4` 和 `--custom-jinja-template examples/chat_templates/gemma4_tool.jinja` 配对 |
| `harmony` | gpt-oss-20b / -120b | Dynamo-only | Harmony channel 格式 |
| `hermes` | Qwen2.5-\*, QwQ-32B, Qwen3-Instruct, Qwen3-Think, NousHermes-2/3 | vLLM: `qwen2_5`; SGLang: `qwen25`（用于 Qwen 模型） | `<tool_call>` JSON |
| `phi4` | Phi-4, Phi-4-mini, Phi-4-mini-reasoning | vLLM: `phi4_mini_json` | `functools[...]` JSON |
| `pythonic` | Llama 4 (Scout / Maverick) | | Python 列表式工具语法 |
| `llama3_json` | Llama 3 / 3.1 / 3.2 / 3.3 Instruct | | `<\|python_tag\|>` 工具语法 |
| `mistral` | Mistral / Mixtral / Mistral-Nemo, Magistral | | `[TOOL_CALLS]...[/TOOL_CALLS]` |
| `jamba` | Jamba 1.5 / 1.6 / 1.7 | Dynamo-only | `<tool_calls>` JSON |
| `default` | *(fallback)* | Dynamo-only | 空 JSON 配置（无 start/end token）。生产环境请优先使用模型专用解析器。 |

<Tip>
对于 Kimi K2.5 thinking 模型，将 `--dyn-tool-call-parser kimi_k2` 与 [Reasoning Parsing (Dynamo)](/dynamo/dev/user-guides/parsing/reasoning-parsing-dynamo) 中的 `--dyn-reasoning-parser kimi_k25` 配对，以便从同一响应中正确解析 `<think>` 块和工具调用。
</Tip>

## 示例

### 启动 Dynamo Frontend 和 Backend

```bash
# 启动后端 worker（或 dynamo.vllm）
python -m dynamo.sglang --model Qwen/Qwen3.5-4B --dyn-tool-call-parser qwen3_coder --dyn-reasoning-parser qwen3

# 启动 frontend
python -m dynamo.frontend
```

### 工具调用请求示例

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

Dynamo 会从模型输出中解析出工具调用，并在响应中以兼容 OpenAI 的 `tool_calls` 形式呈现。

<Tip>
如果工具调用返回结果不正确，请向单个复现请求添加 `"logprobs": true` 并分享响应。有关报告问题时需要捕获和包含的内容，请参阅
[工具调用故障排查](/dynamo/dev/user-guides/parsing/troubleshooting-tool-calls)。
</Tip>

## 可选：结构化标签（structural tags）

你可以启用 **xgrammar 结构化标签**，让引导式解码在 token 粒度上匹配解析器的工具调用格式。参阅 [Structural tag (guided decoding for tool calls)](https://github.com/ai-dynamo/dynamo/blob/869348bb7bd1501deebf6f7f0ab69e1cd69acaba/docs/tool-calling/structural-tag.md)。
