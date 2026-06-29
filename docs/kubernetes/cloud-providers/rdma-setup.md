---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RDMA Setup
subtitle: When Dynamo needs RDMA and where to set it up for your platform
---

Remote Direct Memory Access (RDMA) lets a network adapter move data straight between the memory of two machines, and with GPUDirect RDMA straight between their GPUs, without copying through the CPU, kernel, or TCP/IP stack. NVIDIA Dynamo uses RDMA to transfer KV cache between workers in disaggregated serving. This page explains when you need it and links to the per-platform setup.

## What RDMA Is

RDMA offloads data movement to the NIC, so one host writes directly into another host's memory at near-wire speed and microsecond latency. Dynamo reaches RDMA through NIXL, which transfers KV cache over either UCX or libfabric. Three fabrics provide it:

- **InfiniBand** — dedicated RDMA fabric, common on-premises and on Azure ND-series VMs.
- **RoCE** (RDMA over Converged Ethernet) — RDMA carried over an Ethernet fabric.
- **AWS EFA** (Elastic Fabric Adapter) — the only RDMA fabric on AWS; InfiniBand and RoCE are not offered there.

With GPUDirect RDMA enabled, the NIC reads and writes GPU memory directly, so a KV cache transfer never stages through host memory.

## When You Need It

Dynamo needs RDMA for **disaggregated serving**, where prefill workers generate KV cache and hand it to decode workers. Each handoff is a large GPU-to-GPU transfer, and the transport decides whether it takes milliseconds or seconds.

- **Across nodes** — NVLink does not span nodes, so an RDMA fabric is the only fast path between a prefill worker and a decode worker on different machines.
- **Same node, different pods** — Kubernetes process isolation and GPU partitioning block NVLink between pods, so even co-located prefill and decode workers transfer over the NIC. Use RDMA here too.
- **The alternative is TCP over Ethernet**, which is 200-500x slower for this transfer: roughly 98s Time To First Token (TTFT) on TCP versus 200-500ms with RDMA.

Aggregated deployments run prefill and decode in one worker, transfer no KV cache between workers, and do not need RDMA.

For the full transport comparison (NVLink, InfiniBand, RoCE, and TCP), UCX and libfabric tuning, and GPUDirect RDMA diagnostics, see the [Disaggregated Communication Guide](../disagg-communication-guide.md).

## Set It Up for Your Platform

Pick the guide that matches your fabric:

| Platform | Fabric | Setup guide |
|----------|--------|-------------|
| Azure (AKS) | InfiniBand | [RDMA / InfiniBand on AKS](aks/rdma-infiniband.md) |
| AWS (EKS) | EFA | [EFA (RDMA over AWS Fabric) on EKS](eks/efa.md) |
| On-premises / bare metal | InfiniBand or RoCE | [Disaggregated Communication Guide](../disagg-communication-guide.md) |

Whatever the fabric, the building blocks are the same: an RDMA-capable NIC, a Kubernetes device plugin that advertises the NIC as a schedulable resource (such as `rdma/hca_shared_devices_a` or `vpc.amazonaws.com/efa`), the GPU Operator with GPUDirect RDMA enabled, and worker pods that request the RDMA resource.

## See Also

- [Disaggregated Communication Guide](../disagg-communication-guide.md) — transport options, UCX and libfabric configuration, and performance expectations
- [Multinode Orchestration](../multinode-installation.md) — gang scheduling for workloads that span nodes
- [Disaggregated Serving](../../design-docs/disagg-serving.md) — the architecture that relies on KV cache transfer
