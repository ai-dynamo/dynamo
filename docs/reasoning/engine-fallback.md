---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning Parsing (Engine Fallback)
subtitle: Use upstream vLLM or SGLang reasoning parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a reasoning parser for your model, fall
back to the upstream engine's parser via a **chat-processor swap**, which
keeps frontend tokenization and KV routing.

For Dynamo-native parsers, see [Reasoning Parsing (Dynamo)](dynamo.md). For
the equivalent tool-call fallback, see
[Tool Call Parsing (Engine Fallback)](../tool-calling/engine-fallback.md).

> [!WARNING]
> **Known Issue:** Engine-fallback reasoning parsing does not currently work
> with [disaggregated serving](../features/disaggregated-serving/README.md).
> Use the [Dynamo-native reasoning parser](dynamo.md) for disaggregated
> deployments.

## Configurations

| | Frontend flags | Worker flags | KV routing | Notes |
|---|---|---|---|---|
| **vLLM chat processor** | `--dyn-chat-processor vllm --reasoning-parser <name>` | *(none)* | Yes | Parsing runs in vLLM's Python preprocessor. See [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md). |
| **SGLang chat processor** | `--dyn-chat-processor sglang --reasoning-parser <name>` | *(none)* | Yes | Parsing runs in SGLang's Python preprocessor. See [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md). |

Upstream parser names come from the engine's registry and may differ from
Dynamo's name for the same model (e.g., vLLM's `nemotron_v3` vs Dynamo's
`nemotron3`). They are pinned to the engine version shipped in the Dynamo
container.

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

- [Reasoning Parsing (Dynamo)](dynamo.md) -- Dynamo-native parsers and common pairings
- [Tool Call Parsing (Engine Fallback)](../tool-calling/engine-fallback.md) -- Equivalent fallback for tool-call parsers
- [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
