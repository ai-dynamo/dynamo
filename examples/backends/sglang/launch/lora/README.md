<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA with SGLang Backend

This directory holds local launch scripts for serving LoRA adapters with SGLang from S3-compatible
storage through MinIO.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md).
The SGLang steps are under
[Text LoRA with SGLang](../../../../../docs/fern/pages/cli/operations/lora-adapter-walkthrough.md#text-lora-with-sglang).

## Contents

| Path | Description |
|---|---|
| `setup_minio.sh` | Start MinIO in Docker, download a LoRA from Hugging Face Hub, and upload it to MinIO |
| `agg_lora.sh` | Launch the frontend and a LoRA-enabled SGLang worker |
| `agg_lora_router.sh` | Launch two LoRA-enabled SGLang workers behind the KV router (2 GPUs) |
