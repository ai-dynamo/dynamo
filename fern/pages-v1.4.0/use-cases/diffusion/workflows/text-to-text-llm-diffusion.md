---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Text (LLM Diffusion)
subtitle: Generate text through iterative refinement with SGLang diffusion language models
---

Diffusion Language Models generate text through iterative refinement rather than autoregressive token-by-token generation. The model starts with masked tokens and progressively replaces them with predictions, refining low-confidence tokens each step. See the [Diffusion Overview](../overview.md) for backend setup.

LLM diffusion is auto-detected: when `--dllm-algorithm` is set, the worker automatically uses `DiffusionWorkerHandler` without needing a separate flag. For more details on diffusion algorithms, see the [SGLang Diffusion Language Models documentation](https://docs.sglang.io/docs/supported-models/diffusion_language_models).

## Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/diffusion_llada.sh
```

See the [launch script](https://github.com/ai-dynamo/dynamo/tree/v1.4.0/examples/backends/sglang/launch/diffusion_llada.sh) for configuration options.

## Generate Text

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini-preview",
    "messages": [{"role": "user", "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

## See Also

- [Diffusion Overview](../overview.md)
- [SGLang Examples](../../../recipes/cli-templates/sglang.mdx)
- [SGLang Diffusion LMs (upstream)](https://docs.sglang.io/docs/supported-models/diffusion_language_models)
