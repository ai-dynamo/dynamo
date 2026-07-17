---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Introduction
sidebar-title: Introduction
description: Run and operate Dynamo locally with the CLI — no Kubernetes required
---

Dynamo is an open-source, high-throughput, low-latency inference framework for serving generative AI workloads. It is Kubernetes-native for production, but the same frontend, router, and worker stack also runs on a single machine or VM through the CLI — ideal for evaluation, development, and incremental adoption.

This **CLI Guide** covers installing, deploying, and operating Dynamo locally without Kubernetes. For the production, cluster-based path, see the [Kubernetes Guide](../kubernetes/README.md).

## What you can do from the CLI

- **Install** Dynamo from a container or PyPI, or build it from source.
- **Serve a model** with an OpenAI-compatible endpoint using any supported backend (vLLM, SGLang, TensorRT-LLM).
- **Operate** your deployment with local observability — metrics, logging, tracing, and health checks.

## Get started

Head to the [Quickstart](../getting-started/quickstart.mdx) to get an OpenAI-compatible endpoint running in about 5 minutes, then see [Installation](../getting-started/local-installation.md) for the full walkthrough.
