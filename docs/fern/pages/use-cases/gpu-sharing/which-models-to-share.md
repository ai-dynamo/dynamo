---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Which Models to Share a GPU
subtitle: Workloads that fit MPS and GPU fractions, and why the models must be distinct rather than replicas
---

GPU fractions pays off when several **distinct** models each have too little traffic to keep a GPU busy. It does not pay off for replicas of one model. Replicas belong in one engine, or on whole GPUs of their own. This page explains the difference and lists the workloads that fit.

## Distinct Models, Not Replicas

![Three panels. Two replicas on one GPU, marked avoid: the same model X weights are stored twice, read twice per step, and each copy batches only its own request. One engine for the same traffic, marked do this for replicas: one copy of model X's weights batches both requests and reads the weights once per step; add whole-GPU replicas as traffic grows. Two distinct models on one GPU, marked good fit for MPS: model A and model B have different weights, so they can't share a batch, and alone each leaves the GPU mostly idle, so MPS gives each a capped share.](../../../assets/img/gpu-sharing-replicas-vs-distinct.svg)

On every decode step, an engine reads its model's full weights from GPU memory, so at low and moderate load decode time is set mostly by memory bandwidth. Whether sharing helps depends on whether the models on the GPU have the same weights:

- **Replicas of one model.** Two replicas on one GPU store the same weights twice and read them twice on every step. One engine serving the same requests holds one copy, batches the requests together, and reads the weights once per step for all of them. Packing replicas onto one GPU spends memory and bandwidth for no gain. To serve more traffic for one model, give its engine more requests to batch, and add whole-GPU replicas behind the Dynamo router when it runs out of room.
- **Distinct models.** Two models with different weights can't share a batch, so no single engine can serve both. If each model alone uses a small part of a GPU, the only way to recover the idle part is to run both models on the same GPU. That is what MPS and GPU fractions are for.

Even distinct models pay a cost for sharing. Each model still reads its own weights on every step, and MPS doesn't divide memory bandwidth between them. In the [GPU Sharing](overview.mdx) benchmark, two distinct Qwen3-4B deployments sharing each A100 kept 63.5% of a dedicated GPU's throughput with one request per model, rising to 87.9% at 16 requests per model. That is the cost of serving two models on one GPU instead of two GPUs.

> [!TIP]
> Fine-tuned variants trained as LoRA adapters of one base model are replicas for this purpose: they share the base weights. Serve them from one engine with [LoRA adapters](../../kubernetes/operations/lora-adapters.md) instead of packing a copy of the base model per adapter.

## Workloads That Fit

These workloads have several distinct models whose combined demand fits in one GPU:

| Workload | Why it fits |
|---|---|
| A long tail of models, each with light or intermittent traffic, such as full fine-tunes per customer, internal tools, or language-specific models | Each model would leave a dedicated GPU mostly idle, and their weights differ, so they can't share an engine |
| Supporting models next to an application's main model: embedding, reranking, guardrail and safety classifiers, small routing models | They are small, distinct, and lightly loaded, and they can fill the share a larger model leaves |
| Several model families behind one product, such as chat, code, SQL, and summarization | Each sees part of the product's traffic, and none needs a whole GPU |
| Development, staging, and evaluation environments with many candidate models | Utilization is low and bursty, and GPU count matters more than peak throughput |
| One large tensor-parallel model held to a share of each of its GPUs, with small models in the rest | A model that spans GPUs for memory may not need all of their compute; see [Per-GPU Compute and Memory Limits with MPS](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/mps-per-gpu-limits.md) |

## Workloads That Don't Fit

| Workload | Better choice |
|---|---|
| One model with enough traffic to keep a GPU busy | Dedicated GPUs, scaled with whole-GPU replicas |
| Replicas of one model, or LoRA adapters of one base model | One engine per base model, with batching or multi-LoRA serving |
| A model whose weights and KV cache need most of the GPU's memory | Dedicated GPUs; there is no room for a neighbor |
| Strict tail-latency targets that can't absorb a neighbor's memory bandwidth or power draw | Dedicated GPUs, or MIG, which partitions memory bandwidth |
| Tenants that don't trust each other, or that must not share a fault domain | MIG, or separate GPUs |

## Sizing Each Share

Under GPU fractions, each model asks for two things: a share of the SMs and an amount of memory. Size them from the model and its traffic, not from an even split.

- **Memory:** the model's weights, plus the engine's runtime overhead, plus enough KV cache for the concurrency the model must serve.
- **Compute:** the smallest share at which the model still meets its latency targets at its expected traffic. Measure it: run the model at a few shares, with a neighbor loading the rest of the GPU, and record its time to first token (TTFT) and inter-token latency (ITL).

Then pack models while two budgets hold on each GPU: the SM shares add up to at most 1, and the memory requests fit in the GPU's memory. Memory bandwidth, the L2 cache, and power are not budgeted by MPS, so measure each model next to a busy neighbor rather than alone, and leave headroom.

> [!NOTE]
> A share measured on an idle GPU is optimistic. When every model on the GPU is busy, the GPU can reach its power limit and lower its clocks, and every model's share runs slower. Measure each share with the rest of the GPU loaded.

## Next Steps

- [Example MPS Architecture](example-mps-architecture.md): these sizing rules applied to a two-node cluster.
- [Choosing a GPU Sharing Approach](choosing-a-gpu-sharing-approach.md)
