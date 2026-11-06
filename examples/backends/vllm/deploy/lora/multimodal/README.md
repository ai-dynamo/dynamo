<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multimodal LoRA Deployment on Kubernetes

This directory holds Kubernetes manifests for vision-language (`Qwen/Qwen3-VL-2B-Instruct`) vLLM
inference with LoRA adapters loaded from S3-compatible storage backed by MinIO.

For prerequisites, setup, usage, and troubleshooting, see
[LoRA Adapter Walkthrough](../../../../../../docs/fern/pages/kubernetes/operations/lora-adapter-walkthrough.md).
The vision-language changes are under
[Vision-Language Differences](../../../../../../docs/fern/pages/kubernetes/operations/lora-adapter-walkthrough.md#vision-language-differences).

## Contents

| Path | Description |
|---|---|
| `agg_qwen_lora.yaml` | DynamoGraphDeployment for multimodal vLLM with LoRA support |
| `minio-secret.yaml` | Kubernetes Secret for MinIO credentials |
| `sync-lora-job.yaml` | Job to download a vision LoRA from Hugging Face Hub and upload it to MinIO |
| `lora-model.yaml` | DynamoModel for registering the vision LoRA adapter |
