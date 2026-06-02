---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Call Parsing (Engine Fallback)
subtitle: Use upstream vLLM or SGLang tool-call parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a tool-call parser for your model, fall
back to the upstream engine's parser via a **chat-processor swap**, which
keeps frontend tokenization and KV routing.

This is the **engine-fallback** path. For the Dynamo-native default, see [Tool Call Parsing (Dynamo)](dynamo.md).

> [!IMPORTANT]
> How `--dyn-chat-processor` combines with the parser flags — and which combinations are invalid (engine fallback does **not** support disaggregated serving or TRT-LLM yet) — is documented once in [Parser Configuration](../parser-configuration.md). Read that first; this page covers only the tool-call specifics.

## Configuration

Engine fallback selects the parser with the **frontend** flag `--tool-call-parser <name>` (the engine's own flag), paired with `--dyn-chat-processor vllm` or `sglang`. This is distinct from the Dynamo-native `--dyn-tool-call-parser`, and the accepted names come from the engine's registry — they may differ from Dynamo's (e.g. SGLang `deepseekv3` vs Dynamo `deepseek_v3`).

## Examples

```bash
# vLLM chat processor
python -m dynamo.vllm ...
python -m dynamo.frontend --dyn-chat-processor vllm --tool-call-parser hermes

# SGLang chat processor
python -m dynamo.sglang ...
python -m dynamo.frontend --dyn-chat-processor sglang --tool-call-parser kimi_k2
```

> [!TIP]
> If a tool call comes back wrong, add `"logprobs": true` to a single repro
> request and share the response. See
> [Troubleshooting Tool Calls](troubleshooting.md) for what to capture and
> include when reporting an issue.

## See Also

- [Parser Configuration](../parser-configuration.md) -- how the chat-processor and parser flags combine, and which combinations are invalid (start here)
- [Troubleshooting Tool Calls](troubleshooting.md) -- capture raw model output with `logprobs` so tool-call issues can be localized
- [Tool Call Parsing (Dynamo)](dynamo.md) -- Dynamo-native parsers and request examples
- [Reasoning Parsing (Engine Fallback)](../reasoning/engine-fallback.md) -- Equivalent fallback for reasoning
- [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
