---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parser Engine Fallback
subtitle: Use upstream vLLM or SGLang tool-call and reasoning parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a tool-call or reasoning parser for your model, fall back to the upstream engine's parser via a **chat-processor swap**, which keeps frontend tokenization and KV routing.

For the Dynamo-native default path, see [Tool Call Parsing (Dynamo)](tool-calling/dynamo.md) and [Reasoning Parsing (Dynamo)](reasoning/dynamo.md).

> [!IMPORTANT]
> How `--dyn-chat-processor` combines with the parser flags — and which combinations are invalid (engine fallback does **not** support disaggregated serving, and TRT-LLM support is a work in progress) — is documented once in [Parser Configuration](parser-configuration.md). Read that first; this page covers only the engine-fallback specifics.

## Configuration

Engine fallback runs parsing in the engine's own Python frontend. Select it with `--dyn-chat-processor vllm` or `sglang`, then name the parser with the engine's **frontend** flags:

- `--tool-call-parser <name>` — the engine's tool-call parser
- `--reasoning-parser <name>` — the engine's reasoning parser

These are distinct from the Dynamo-native `--dyn-tool-call-parser` / `--dyn-reasoning-parser` (which go on the worker). The accepted names come from the engine's registry and may differ from Dynamo's — e.g. vLLM `nemotron_v3` vs Dynamo `nemotron3`, SGLang `deepseekv3` vs Dynamo `deepseek_v3`.

## Examples

```bash
# vLLM chat processor — frontend carries the parser flags, then launch the worker:
python -m dynamo.frontend --dyn-chat-processor vllm   --tool-call-parser hermes --reasoning-parser qwen3
python -m dynamo.vllm   --model Qwen/Qwen3-0.6B

# SGLang chat processor
python -m dynamo.frontend --dyn-chat-processor sglang --tool-call-parser qwen25 --reasoning-parser qwen3
python -m dynamo.sglang --model Qwen/Qwen3-0.6B
```

> [!TIP]
> If a tool call or reasoning split comes back wrong, add `"logprobs": true` to a single repro request and share the response. See [Troubleshooting Tool Calls](tool-calling/troubleshooting.md) for what to capture.

## See Also

- [Parser Configuration](parser-configuration.md) -- how the chat-processor and parser flags combine, and which combinations are invalid (start here)
- [Tool Call Parsing (Dynamo)](tool-calling/dynamo.md) -- Dynamo-native tool-call parser names
- [Reasoning Parsing (Dynamo)](reasoning/dynamo.md) -- Dynamo-native reasoning parser names
- [vLLM Chat Processor](backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](components/frontend/configuration.md) -- Full CLI flag reference
