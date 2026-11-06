<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA Deployment on Kubernetes

This directory holds Kubernetes manifests for LoRA-enabled vLLM inference, loading adapters directly
from Hugging Face Hub or through S3-compatible storage backed by MinIO.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../docs/fern/pages/kubernetes/operations/lora-adapter-walkthrough.md).

## Contents

| Path | Description |
|---|---|
| `agg_lora.yaml` | DynamoGraphDeployment for vLLM with LoRA support, loading adapters from MinIO |
| `agg_lora_hf.yaml` | DynamoGraphDeployment and DynamoModel for direct Hugging Face Hub loading |
| `minio-secret.yaml` | Kubernetes Secret for MinIO credentials |
| `sync-lora-job.yaml` | Job to download a LoRA from Hugging Face Hub and upload it to MinIO |
| `lora-model.yaml` | DynamoModel for registering the LoRA adapter |
| `multimodal/` | Vision-language (Qwen3-VL) LoRA manifests |
