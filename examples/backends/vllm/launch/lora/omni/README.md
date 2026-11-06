<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Omni LoRA (Image) with vLLM Backend

This directory holds local scripts for vLLM-Omni image generation with dynamic LoRA load and unload,
running the Dynamo frontend and an Omni worker in aggregated mode.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md).
The image generation steps are under
[Image Generation LoRA with vLLM-Omni](../../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md#image-generation-lora-with-vllm-omni).

## Contents

| Path | Description |
|---|---|
| `omni_lora_agg.sh` | Launch the frontend and an Omni worker with `--enable-lora` |
| `validate_omni_lora_agg.sh` | Validate the Omni LoRA endpoints and generation flow |
