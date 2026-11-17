---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: MPS Fault Isolation
subtitle: Which GPU faults reach the other models on a shared GPU, measured for each MPS configuration
---

A GPU fault is an error a kernel can't recover from, such as a write to an address the process doesn't own. Without MPS, each process has its own context, so a fault stops only the process that caused it. Under MPS, the clients run through the server, and whether a fault reaches the other models depends on two things: how the server separates the models, and what kind of fault it is.

## What Each Configuration Contains

![What a fault in model B reaches. Namespace per model, the GPU fractions layout: concurrent; a memory fault, trap, or breakpoint reaches every model; a device-side assert stays with B. SM partition per model: concurrent; a memory fault stays with B's partition; a trap or breakpoint reaches every model; an assert stays with B. MPS server per model: the models take turns; every fault stays with B. MIG, for comparison and not measured: concurrent; every fault stays with B.](../../../../../assets/img/gpu-sharing-mps-faults.svg)

The figure comes from a test on one GPU: two models kept the GPU busy, and a third client, sharing model B's namespace, partition, or server, triggered a fault on purpose. Read it one row at a time:

- **One namespace per model.** This is the GPU fractions layout. A namespace gives a model its own limits, but not its own fault domain. When model B hit an illegal address, model A received the same error at the same moment, and every client on the GPU had to exit.
- **One SM partition per model.** A static SM partition gives a model its own set of SMs. When model B hit an illegal address, the MPS server stopped only the clients in B's partition, and model A kept serving without an error. The models still ran at the same time. This held with MPS v2 and v3, and whether A and B were in the same namespace or different ones.
- **One server per model.** Two MPS servers on the same GPU kept every fault inside one server, including traps and breakpoints. But the GPU took turns between the servers, as it does without MPS, which gives up the reason to use MPS.

## Traps, Breakpoints, and Asserts

Two kinds of fault behave differently from the rest:

- **Traps and breakpoints reach every model on the server.** A kernel that executes a trap instruction, such as `__trap()`, or a breakpoint makes the MPS server reset the GPU, with or without partitions. Every model on that server fails, so with one server per GPU, every model on the GPU fails. Traps are rare in serving, and usually come from a library rather than your own code. For example, the DeepEP kernels that exchange tokens between GPUs in an expert-parallel MoE model call `trap()` when a peer GPU doesn't respond in time, and the engine reports `unspecified launch failure`. With partitions, new clients also failed until the MPS control daemon was restarted.
- **Device-side asserts stay with the model.** A failed `assert` in a kernel, such as the index check behind PyTorch's `device-side assert triggered` error, stopped only the model that asserted, in every configuration. That model still has to restart.

## What Static SM Partitions Cost

Static SM partitions are the only configuration that keeps memory faults inside one model while the models keep running at the same time. They cost flexibility:

- **SMs come in fixed chunks.** A partition is a whole number of chunks of 8 SMs, so sizes are rounded, and SMs left over after the last chunk can go unused. A partition can't lend idle SMs to a busy neighbor, unlike an active thread percentage, which is only a ceiling.
- **The rest stays shared.** Memory bandwidth, the L2 cache, and power are still shared, and models in the same partition share its faults.
- **GPU fractions doesn't use them.** GPU fractions caps each model's compute with an active thread percentage, so the models on each GPU share one fault domain.

[MPS v2 and v3](mps-v2-and-v3.md#static-sm-partitions) shows how to create a partition. For isolation from every kind of fault, use MIG or separate GPUs.

> [!NOTE]
> These results were measured on an RTX PRO 6000 Blackwell GPU with driver 615.71.09, in single-user MPS mode. Fault handling can differ between GPUs and drivers, so repeat the test on your hardware before you depend on it.

## Related Pages

- [How GPU Sharing Works](how-gpu-sharing-works.md): MPS compared with time-slicing, HAMi, and MIG.
- [Operate Shared GPUs](../../../../use-cases/gpu-sharing/operate-shared-gpus.md): how to plan for faults when you place models.
