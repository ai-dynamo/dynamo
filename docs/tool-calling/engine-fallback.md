---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tool Call Parsing (Engine Fallback)
subtitle: Use upstream vLLM or SGLang tool-call parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a tool-call parser for your model, fall
back to the upstream engine's parser via a **chat-processor swap**, which
keeps frontend tokenization and KV routing.

For Dynamo-native parsers, see [Tool Call Parsing (Dynamo)](dynamo.md). For
the equivalent reasoning fallback, see
[Reasoning Parsing (Engine Fallback)](../reasoning/engine-fallback.md).

> [!WARNING]
> **Known Issue:** Engine-fallback tool call parsing does not currently work
> with [disaggregated serving](../features/disaggregated-serving/README.md).
> Use the [Dynamo-native tool call parser](dynamo.md) for disaggregated
> deployments.

## Configurations

| | Frontend flags | Worker flags | KV routing | Notes |
|---|---|---|---|---|
| **vLLM chat processor** | `--dyn-chat-processor vllm --tool-call-parser <name>` | *(none)* | Yes | Parsing runs in vLLM's Python preprocessor. See [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md). |
| **SGLang chat processor** | `--dyn-chat-processor sglang --tool-call-parser <name>` | *(none)* | Yes | Parsing runs in SGLang's Python preprocessor. See [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md). |

Upstream parser names come from the engine's registry and may differ from
Dynamo's name for the same model (e.g., SGLang's `deepseekv3` vs Dynamo's
`deepseek_v3`). They are pinned to the engine version shipped in the Dynamo
container.

## Examples

```bash
# vLLM chat processor
python -m dynamo.vllm ...
python -m dynamo.frontend --dyn-chat-processor vllm --tool-call-parser hermes

# SGLang chat processor
python -m dynamo.sglang ...
python -m dynamo.frontend --dyn-chat-processor sglang --tool-call-parser kimi_k2
```

## See Also

- [Tool Call Parsing (Dynamo)](dynamo.md) -- Dynamo-native parsers and request examples
- [Reasoning Parsing (Engine Fallback)](../reasoning/engine-fallback.md) -- Equivalent fallback for reasoning
- [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
