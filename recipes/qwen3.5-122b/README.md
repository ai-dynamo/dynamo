<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5-122B-A10B Recipes

Recipes for [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) —
122B total / 10B active hybrid MoE (Gated DeltaNet linear attention + MoE with full
attention every 4th layer, 262,144-token context, vision input supported).

Each quantized checkpoint is a self-contained recipe folder with its own
`model-cache/`, `vllm/` profiles and perf notes.

## Variants

| Variant | Checkpoint | Hardware | Profiles |
| ------- | ---------- | -------- | -------- |
| [`fp8/`](fp8/) | [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | H200 | agg (tp1 + MTP), disagg 1P2D |
| [`nvfp4/`](nvfp4/) | [nvidia/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4) | B200 | agg, disagg 1P2D |

Pick the variant that matches your hardware, then follow the README inside it.
