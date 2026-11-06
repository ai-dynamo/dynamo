<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Offline Batch Inference with Dynamo

This directory holds the `dynamo-batch` Helm chart, a sample DGD, and helper scripts that add an
experimental OpenAI Batch lifecycle to a Dynamo frontend.

For prerequisites, deployment steps, and configuration, see
[Offline Batch Inference](../../../docs/fern/pages/kubernetes/model-deployment/offline-batch-inference.md).

## Contents

| Path | Description |
|---|---|
| `dynamo.yaml` | Sample DGD serving `Qwen/Qwen3-0.6B` |
| `profile-external-storage.yaml` | Batch profile for the target DGD, model, and external PostgreSQL, Redis/Valkey, and S3 storage |
| `render_values.py` | Renders a `dynamo-batch` Helm values file from a profile |
| `run_example.py` | Uploads a batch input file, runs a job, and verifies its results |
| `batch-input.jsonl` | Sample batch input with two chat completion requests |
| `chart/` | `dynamo-batch` Helm chart that deploys the batch services for a DGD |
