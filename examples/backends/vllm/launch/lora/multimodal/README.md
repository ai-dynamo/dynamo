<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multimodal LoRA with vLLM Backend

This directory holds local scripts for serving vision-language models with dynamically loadable LoRA
adapters, using Dynamo's aggregated architecture.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md).
The vision-language steps are under
[Vision-Language LoRA with vLLM](../../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md#vision-language-lora-with-vllm).

## Contents

| Path | Description |
|---|---|
| `lora_agg.sh` | Launch the frontend and a LoRA-enabled multimodal vLLM worker (default `Qwen/Qwen3-VL-2B-Instruct`) |
| `validate_lora_agg.sh` | Validate the LoRA endpoints and multimodal inference against a running server |
