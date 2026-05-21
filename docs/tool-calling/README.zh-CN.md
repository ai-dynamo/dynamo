---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Calling
subtitle: Parse tool calls from model output and surface them as OpenAI-compatible tool_calls
---

<p align="left">
  <a href="./README.md" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

Dynamo 可以通过从原始模型输出中解析 tool-call 语法，并将其作为 OpenAI 兼容的
`tool_calls` 暴露在响应中，从而把模型连接到外部工具和服务。工具调用由 chat
completions API 上的 `tool_choice` 和 `tools` 请求参数控制。

在 Dynamo 中解析工具调用有两种方式，具体取决于解析器是位于 Dynamo 自身的注册表中，还是位于上游引擎（vLLM、SGLang）中。

## 选择解析路径

| 路径 | 何时使用 | 页面 |
|------|----------|------|
| **Dynamo** | Dynamo 为该模型的 tool-call 格式提供了 Rust 解析器。延迟最低，也是默认路径。 | [工具调用解析（Dynamo）](dynamo.md) |
| **Engine Fallback** | 使用框架（vLLM 或 SGLang）的实现进行预处理/后处理，包括工具调用和推理解析，以确保与框架行为保持一致。 | [工具调用解析（Engine Fallback）](engine-fallback.md) |

请从 Dynamo 路径开始。只有当 Dynamo 的注册表没有列出适用于你的模型的解析器时，才回退到引擎路径。

## 故障排查

如果工具调用返回结果不正确，请向单个复现请求添加 `logprobs: true` 并分享响应。有关报告问题时需要捕获和包含的内容，请参阅
[工具调用故障排查](troubleshooting.md)。

## 另请参阅

- [工具调用故障排查](troubleshooting.md) -- 使用 `logprobs` 捕获原始模型输出，以便定位工具调用问题。
- [Reasoning](../reasoning/README.md) -- 将 `reasoning_content` 与 chain-of-thought 模型的 assistant content 分离。多个模型需要同时配置工具调用解析器和推理解析器。
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- 完整的 CLI flag 参考。
