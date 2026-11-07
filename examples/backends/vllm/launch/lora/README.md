<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA with vLLM Backend

This directory holds local launch scripts for serving LoRA adapters with vLLM, loading them from
Hugging Face Hub or from S3-compatible storage through MinIO.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md).

## Contents

| Path | Description |
|---|---|
| `setup_minio.sh` | Start MinIO in Docker, download a LoRA from Hugging Face Hub, and upload it to MinIO |
| `agg_lora.sh` | Launch the frontend and a LoRA-enabled vLLM worker that loads adapters from MinIO |
| `agg_lora_hf.sh` | Launch the frontend and a LoRA-enabled vLLM worker that loads the adapter from Hugging Face Hub |
| `agg_lora_router.sh` | Launch two LoRA-enabled vLLM workers behind the KV-aware router (2 GPUs) |
| `multimodal/` | Vision-language LoRA launch and validation scripts |
| `omni/` | vLLM-Omni image generation LoRA launch and validation scripts |
| `xpu/` | Intel XPU variants of `agg_lora.sh` and `agg_lora_router.sh` |
