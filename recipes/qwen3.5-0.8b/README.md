<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-0.8B vLLM Aggregated Recipe

Deploys `Qwen/Qwen3.5-0.8B` on vLLM in aggregated mode against `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0`.

## Deploy

```bash
export NAMESPACE=your-namespace
kubectl apply -n ${NAMESPACE} -f vllm/agg/deploy.yaml
```

The DGD references two PVCs (`model-cache`, `compilation-cache`) with `create: false` -- ensure both exist in the namespace.

## Parsers

- `--dyn-reasoning-parser qwen3` -- extracts `<think>...</think>` into `reasoning_content`.
- `--dyn-tool-call-parser qwen3_coder` -- parses Qwen3-family XML tool calls into structured `tool_calls`.
