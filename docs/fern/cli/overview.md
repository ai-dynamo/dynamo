---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Overview
sidebar-title: Overview
description: Deploy and serve models locally with Dynamo backends
---

Once Dynamo is installed, you can serve a model locally with an OpenAI-compatible endpoint using any supported inference backend. This section covers backend-specific deployment steps and local KV cache offloading.

## Backends

Dynamo supports three inference backends. Each has a deployment guide with the commands to launch a frontend, router, and worker on a single machine:

- **[vLLM](../backends/vllm/vllm-examples.md)** — broad model coverage and a mature feature set.
- **[SGLang](../backends/sglang/sglang-examples.md)** — high-throughput serving with RadixAttention.
- **[TensorRT-LLM](../backends/trtllm/trtllm-examples.md)** — NVIDIA-optimized inference for maximum performance.

Not sure which to pick? See the [Compatibility](../reference/compatibility.mdx) matrix for supported models and features per backend.

## KV cache offloading

Extend effective KV cache capacity by offloading to CPU memory or remote stores. See the KV Cache Offloading pages for [LMCache](../integrations/lmcache-integration.md), [HiCache](../backends/sglang/sglang-hicache.md), [FlexKV](../integrations/flexkv-integration.md), and [custom KV events](../integrations/kv-events-custom-engines.md).

> [!NOTE]
> This overview is a work in progress. Backend deployment guides linked above contain the full, tested instructions.
