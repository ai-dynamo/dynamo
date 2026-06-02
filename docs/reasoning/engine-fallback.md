---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning Parsing (Engine Fallback)
subtitle: Use upstream vLLM or SGLang reasoning parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a reasoning parser for your model, fall
back to the upstream engine's parser via a **chat-processor swap**, which
keeps frontend tokenization and KV routing.

This is the **engine-fallback** path. For the Dynamo-native default, see [Reasoning Parsing (Dynamo)](dynamo.md).

> [!IMPORTANT]
> How `--dyn-chat-processor` combines with the parser flags — and which combinations are invalid (engine fallback does **not** support disaggregated serving or TRT-LLM yet) — is documented once in [Parser Configuration](../parser-configuration.md). Read that first; this page covers only the reasoning specifics.

## Configuration

Engine fallback selects the parser with the **frontend** flag `--reasoning-parser <name>` (the engine's own flag), paired with `--dyn-chat-processor vllm` or `sglang`. This is distinct from the Dynamo-native `--dyn-reasoning-parser`, and the accepted names come from the engine's registry — they may differ from Dynamo's (e.g. vLLM `nemotron_v3` vs Dynamo `nemotron3`).

## Examples

```bash
# vLLM chat processor
python -m dynamo.vllm ...
python -m dynamo.frontend --dyn-chat-processor vllm --reasoning-parser deepseek_r1

# SGLang chat processor
python -m dynamo.sglang ...
python -m dynamo.frontend --dyn-chat-processor sglang --reasoning-parser kimi_k25
```

## See Also

- [Parser Configuration](../parser-configuration.md) -- how the chat-processor and parser flags combine, and which combinations are invalid (start here)
- [Reasoning Parsing (Dynamo)](dynamo.md) -- Dynamo-native parsers and common pairings
- [Tool Call Parsing (Engine Fallback)](../tool-calling/engine-fallback.md) -- Equivalent fallback for tool-call parsers
- [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
