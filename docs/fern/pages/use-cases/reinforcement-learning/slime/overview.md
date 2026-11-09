---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Slime Integration
subtitle: Run Slime external rollouts through Dynamo on Kubernetes
---

[Slime](https://github.com/THUDM/slime) is an RL post-training framework tbuilt around Megatron for training and SGLang for inference. It runs as a RayJob through KubeRay operator on K8s or the Ray CLI for local environments.

This tutorial will cover the Kubernetes use case. The Dynamo integration with Slime brings in the Dynamo operator to serve the rollouts on a DGD with the SGLang backend. The Slime RL rollout benefits from the Dyanmo frontend and KV router. Dynamo includes a Slime recipe for convenience, but this guide will walk through slime GRPO training (Qwen3-0.6B, GSM8K, 10 steps) .

## Training: Kuberay RayJob

Set Up: 
- Installation...
- Prep datea 
....

## Inference

- Deploy Dynamo DGD
