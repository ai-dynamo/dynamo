---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 工具调用解析（Dynamo）
subtitle: 使用 Dynamo 内置的工具调用解析器，将模型连接到外部工具和服务
---

<p align="left">
  <a href="./README.mdx" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

你可以通过工具调用把 Dynamo 连接到外部工具和服务。通过提供一组可用函数，Dynamo 可以为相关函数输出函数参数，你执行这些函数后，即可用外部信息来增强提示。

工具调用由 `tool_choice` 和 `tools` 请求参数控制。

本页介绍默认 Dynamo 原生路径的解析器名称，即由 `dynamo` [chat processor](chat-processors.mdx) 在 worker 上解析工具调用。如果 Dynamo 未列出适用于你的模型的解析器，请改用
[engine fallback](chat-processors.mdx)。

## 启用工具调用

要启用工具调用，请在 DynamoGraphDeployment (DGD) 中后端 **worker** 的 `args:` 里添加 `--dyn-tool-call-parser`，取值为与你的模型匹配的解析器。它的位置如下：

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: ...
spec:
  components:
  - name: SGLangWorker
    type: worker
    ...
    podTemplate:
      spec:
        containers:
        - name: main
          ...
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3.5-4B
          - --served-model-name
          - Qwen/Qwen3.5-4B
          - --dyn-tool-call-parser   # 添加此项以启用工具调用
          - qwen3_coder              # 取值来自下方表格
```

Frontend 无需额外 flag——默认的 `dynamo` chat processor 会为每个后端解析工具调用。

> [!IMPORTANT]
> `--dyn-tool-call-parser` 与默认的 `dynamo` chat processor 配对，放在 **worker** 上。裸的 `--tool-call-parser`（无 `--dyn-` 前缀）是另一个 flag——它驱动 [engine fallback](chat-processors.mdx) 并放在 Frontend 上。不要在默认 processor 下使用裸 flag，也不要把 `--dyn-tool-call-parser` 放在 `vllm`/`sglang` chat processor 上。

该 flag 可用于任意 worker——`dynamo.vllm`、`dynamo.sglang` 或 `dynamo.trtllm`。完整的 worker flag 列表请参阅各后端指南（[vLLM](../backends/vllm/README.md)、[SGLang](../backends/sglang/README.md)、[TensorRT-LLM](../backends/trtllm/README.md)），或以 `--help` 运行 worker。初次编写 DGD？请从 [Deploy with DGD](../kubernetes/dgd-guide.md) 开始。

> [!TIP]
> 如果你的模型默认的 chat template 不支持工具调用，但模型本身支持，可以在 worker 的 `args:` 中添加 `--custom-jinja-template /path/to/template.jinja`（该模板文件需在容器内可读）。

> [!TIP]
> 如果你的模型还会输出需要与正常内容分离的推理内容，请参阅 [Reasoning Parsing (Dynamo)](../reasoning/README.md) 了解支持的 `--dyn-reasoning-parser` 取值。

## 支持的工具调用解析器

将 `--dyn-tool-call-parser` 设置为与你的模型匹配的**解析器名称**（第一列）。以下是它接受的全部取值。

**Upstream name** 列标出 vLLM 或 SGLang 的解析器名称与 Dynamo 不同之处——在使用 `--dyn-chat-processor vllm` 或 `sglang` 时（参阅 [engine fallback](chat-processors.mdx)）尤为相关。upstream 列为空表示同名在各处通用。`Dynamo-only` 表示该格式没有对应的上游解析器。

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

> [!TIP]
> 对于 Kimi K2.5 thinking 模型，将 `--dyn-tool-call-parser kimi_k2` 与 [Reasoning Parsing (Dynamo)](../reasoning/README.md) 中的 `--dyn-reasoning-parser kimi_k25` 配对，以便从同一响应中正确解析 `<think>` 块和工具调用。

## 示例

### 部署 Frontend 和 worker

下面是上述 worker 片段对应的完整可运行 DGD。它在 SGLang 上部署 Qwen3.5-4B 并启用工具调用解析（因该模型同时输出推理内容，故还加了 `--dyn-reasoning-parser`）——将 worker 换成使用相同 `--dyn-*` flag 的 `dynamo.vllm` 或 `dynamo.trtllm` 亦可：

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen35-tools
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: SGLangWorker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3.5-4B
          - --served-model-name
          - Qwen/Qwen3.5-4B
          - --dyn-tool-call-parser
          - qwen3_coder
          - --dyn-reasoning-parser
          - qwen3
```

应用该文件并等待部署就绪：

```bash
kubectl apply -f qwen35-tools.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready dynamographdeployment/qwen35-tools \
  -n ${NAMESPACE} --timeout=600s
```

### 工具调用请求示例

对 Frontend Service 做端口转发，然后发送请求：

```bash
kubectl port-forward svc/qwen35-tools-frontend 8000:8000 -n ${NAMESPACE}
```

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

> [!TIP]
> 如果工具调用返回结果不正确，请向单个复现请求添加 `"logprobs": true` 并分享响应。有关报告问题时需要捕获和包含的内容，请参阅
> [工具调用故障排查](troubleshooting.md)。

## 可选：结构化标签（structural tags）

你可以启用 **xgrammar 结构化标签**，让引导式解码在 token 粒度上匹配解析器的工具调用格式。参阅 [Structural tag (guided decoding for tool calls)](structural-tag.md)。
