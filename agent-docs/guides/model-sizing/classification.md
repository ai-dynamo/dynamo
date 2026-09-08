<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model Classification

Inspect the exact model revision used by the DGD before sizing memory or selecting parallelism. The model structure and
attention type determine weight placement, KV-cache size, and useful parallel strategies.

## Fetching Model Information

For a Hugging Face model, inspect these files at the exact revision being served:

- `https://huggingface.co/<org>/<model>/raw/<revision>/config.json` — architecture, layer dimensions, context limit,
  attention layout, and MoE settings.
- `https://huggingface.co/<org>/<model>/raw/<revision>/hf_quant_config.json` — ModelOpt quantization details when
  present.
- `https://huggingface.co/<org>/<model>/raw/<revision>/model.safetensors.index.json` — weight shards and tensor names
  when present.

Record the model revision and relevant file hashes. For a local model, inspect the equivalent files in its model
directory. Do not infer architecture, parameter count, or precision from the model name.

## Model Architecture Classification

Classify the model on two primary axes:

- **Structure**: dense or mixture-of-experts (MoE).
- **Attention**: multi-head attention (MHA), grouped-query attention (GQA), or multi-head latent attention (MLA).

Also record total and active parameters, layer count, hidden size, attention heads, KV heads, head dimension, expert
count, experts selected per token, maximum context length, modality, stored weight type, and quantization metadata.

### Architecture Classes

| Architecture class | Model structure | Attention type | Key traits |
| --- | --- | --- | --- |
| **MoE_MLA** | MoE | MLA | Compressed KV; usually more KV-cache headroom than GQA |
| **MoE_GQA** | MoE | GQA | Larger KV footprint than MLA |
| **Dense_GQA** | Dense | GQA | No MoE params; straightforward TP; larger KV footprint than MLA |
| **Dense_MHA** | Dense | MHA | Full-size KV per head; legacy models |
| **Encoder-decoder** | Varies | Varies | Separate encoder and decoder; cross-attention KV cache |
| **Multimodal** | Varies | Varies | Vision encoder + LLM decoder; extra memory for vision features |

### Why Attention Architecture Matters

| Property | GQA (many KV heads) | MLA (compressed latent KV) |
| --- | --- | --- |
| Per-token KV cache size | Large | Small (compressed) |
| Safe `free_gpu_memory_fraction` | Lower (0.75-0.85) | Higher (0.80-0.90) |
| Sensitivity to fraction tuning | High | Low |

**Rule:** classify attention architecture before tuning KV-cache fractions. MLA-safe fractions may still OOM on GQA
models.
