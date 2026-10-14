---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Many-Model Serving
subtitle: Pack several small models onto shared GPUs on Kubernetes
sidebar-title: Overview
---

Serving many small models with one model per GPU leaves most of each GPU idle at
light load. This section shows how to run two models per GPU on Kubernetes with
Dynamo, and compares two ways to isolate them against a dedicated-GPU baseline.

## Options Compared

| Configuration | Models per GPU | Isolation | Maturity |
| --- | --- | --- | --- |
| Baseline | 1 | Dedicated GPU | Standard |
| KAI + HAMi | 2 | Memory cap only; the CUDA driver time-slices compute between the two models | Released components |
| KAI + GPU fractions | 2 | Memory cap plus a fixed share of the GPU's streaming multiprocessors (SMs) through CUDA MPS; both models run concurrently | Experimental: built from open upstream pull requests, needs driver r615 and GPU Operator v26.7.1 or later |

On one 8× A100 40GB node serving Qwen3-4B, GPU fractions kept 60–88% of a
dedicated GPU's throughput, depending on load, against 45–73% for HAMi. At light
per-model load, serving 16 models on one node with GPU fractions cost up to 23%
less per output token than two nodes of dedicated GPUs. See
[GPU Sharing Benchmark Results](gpu-sharing-results.mdx) for the full tables and
how to read them.

## Terms

| Term | Meaning |
| --- | --- |
| DGD | `DynamoGraphDeployment`, the Kubernetes resource that deploys one Dynamo model |
| KAI-Scheduler | A Kubernetes scheduler that can place pods on a fraction of a GPU |
| HAMi | A GPU virtualization layer that enforces per-pod GPU memory limits |
| MPS | CUDA Multi-Process Service, which lets processes share a GPU with a fixed SM share and memory limit |
| gpu-fractioning | The KAI-Scheduler component that runs MPS per GPU and injects each pod's SM and memory limits |

## Reading Order

1. [GPU Sharing Benchmark Results](gpu-sharing-results.mdx): what was measured and how to read it.
2. [Common Setup](common-setup.md): install the Dynamo platform and cache the model once.
3. Deploy one or more experiments:
   - [Baseline](deploy-baseline.md)
   - [KAI + HAMi](deploy-kai-hami.md)
   - KAI + GPU fractions: [Build the KAI + GPU Fractions Stack](build-gpu-fractioning-forks.md), then [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md)
4. [Per-GPU Compute and Memory Limits with MPS](mps-per-gpu-limits.md): the MPS mechanism the GPU fractions stack automates, run by hand with Docker.
