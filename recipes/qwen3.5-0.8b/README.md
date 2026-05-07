<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-0.8B vLLM Aggregated Recipe

Smallest dense multimodal Qwen3.5 variant (`Qwen/Qwen3.5-0.8B`) deployed on a single H100 with vLLM, reasoning parser, and tool-call parser wired in.

## Available Configurations

| Configuration | GPUs | Backend | Mode | Description |
|--------------|------|---------|------|-------------|
| [**vllm/agg**](vllm/agg/) | 1x H100 | vLLM | Aggregated | TP=1, multimodal, qwen3 reasoning + qwen3_coder tool parser |

## Prerequisites

1. **Dynamo Platform installed** -- See the [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **1x H100 80GB GPU** (model is ~2 GB in BF16; 80 GB is generous and accommodates the multimodal vision tower plus a 32K context KV cache)
3. **HuggingFace token** with access to the Qwen org (model is currently public, but the secret is wired uniformly across recipes):
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

```bash
export NAMESPACE=your-namespace

# 1. Provision PVCs (edit storageClassName for your cluster)
kubectl apply -n ${NAMESPACE} -f model-cache/model-cache.yaml

# 2. Pre-download weights to the PVC
kubectl apply -n ${NAMESPACE} -f model-cache/model-download.yaml

# 3. Deploy the graph
kubectl apply -n ${NAMESPACE} -f vllm/agg/deploy.yaml
```

Send a chat request once the frontend is healthy:

```bash
curl ${DYNAMO_FRONTEND}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "stream": false,
    "max_tokens": 256
  }' | jq
```

## Parser Notes

The recipe sets:

- `--dyn-reasoning-parser qwen3` -- extracts `<think>...</think>` chain-of-thought into a separate `reasoning_content` field on responses. This is the same parser used for Qwen3; the `preprocessor.rs` chat-template hook recognises Qwen3.5's `<think>\n` opener.
- `--dyn-tool-call-parser qwen3_coder` -- parses Qwen3-family tool-call blocks into structured `tool_calls`. The `qwen3_coder` parser handles both single and parallel tool calls.

## Architecture Note

`Qwen/Qwen3.5-0.8B` registers in vLLM 0.19.1 as `Qwen3_5ForConditionalGeneration` in the multimodal model registry (`vllm/model_executor/models/registry.py`, module `qwen3_5`). Hence `--enable-multimodal` -- removing it will fall back to text-only and will likely fail to instantiate the vision tower correctly.

Dynamo's `model_card.rs` applies a Qwen3.5-specific EOS workaround: the model ships with `text_config.eos_token_id = 248044` (`<|endoftext|>`), which is the base EOS rather than the chat-tuned `<|im_end|>`. The fix-up is automatic when the frontend recognises `model_type: qwen3_5`.

## Validation (2026-05-07)

End-to-end deploy validated on `cl-h100` (1x H100-80GB, computelab) using `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0`:

```
2026-05-07T21:16:07  INFO model.__post_init__: Resolved architecture: Qwen3_5ForConditionalGeneration
2026-05-07T21:17:18  INFO _core: Registered base model 'Qwen/Qwen3.5-0.8B' MDC
2026-05-07T21:17:18  INFO main.get_engine_cache_info: Cache config: num_gpu_blocks=9920, block_size=544
2026-05-07T21:17:20  INFO dynamo_llm::http::service::service_v2: chat endpoints enabled
```

A real chat completion (`POST /v1/chat/completions`, 33 prompt tokens, 256 completion tokens) returned in 1987 ms with `reasoning_content` field present (qwen3 reasoning parser plumbed correctly).

### Notes from the smoke test

- **Use image tag `1.1.0`, not `1.0.0`.** The 1.0.0 image shipped with `vllm==0.16.0` and `transformers==4.57.6` -- both pre-date Qwen3.5's `model_type: qwen3_5`, so a deploy against 1.0.0 hits a Transformers `ValidationError` before the engine ever instantiates. The 1.1.0 image (released 2026-05-04) bumps both pins and resolves the architecture cleanly.
- **vLLM CLI rename.** The flag formerly called `--disable-log-requests` was renamed to `--no-enable-log-requests` in the vLLM version shipped with 1.1.0. The recipe uses the new spelling. Older Dynamo recipes still using `--disable-log-requests` will fail with `unrecognized arguments` on the new image.
- **Qwen3.5 is hybrid Mamba + Attention.** vLLM auto-sets `Mamba cache mode = align` and pads the mamba page size by 2.64% to keep attention and mamba page sizes equal. No recipe-side knobs needed; same family pattern as Nemotron-3-Super.
