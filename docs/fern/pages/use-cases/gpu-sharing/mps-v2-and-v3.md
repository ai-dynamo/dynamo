---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: MPS v2 and v3
subtitle: How the MPS v3 interface changes where limits live, and why per-model GPU fractions need it
---

CUDA Multi-Process Service (MPS) has two control interfaces. The v2 interface, now called the legacy interface, runs one MPS server per Linux user, and each client sets its own limits. The v3 interface runs many named servers, each holding namespaces, and the limits live in the namespace on the server. Packing distinct models with a different size for each pod depends on v3. This page explains what changed. [How GPU Sharing Works](how-gpu-sharing-works.md) covers MPS itself.

![Left, MPS v2: one control daemon per Linux user runs one MPS server with default limits; client A sets 60% of SMs and 30 GB in its environment, and client B sets 40% and 20 GB, and each can only lower the server's defaults. On Kubernetes, the NVIDIA device plugin's MPS mode cuts a GPU into equal replicas. Right, MPS v3: one control daemon per node runs a server per GPU. Server gpu-0 holds three namespaces, chat-8b at 50% of SMs and 32 GB, code-7b at 30% and 24 GB, and guard-1b at 20% and 8 GB, each serving one pod. Each pod sees only its own namespace's pipe directory. V3 also adds cgroup memory limits per container per GPU, several named servers in one daemon, a TOML configuration file, and client priority per namespace.](../../../assets/img/gpu-sharing-mps-v2-v3.svg)

## MPS v2: Limits Travel With the Client

In v2, the control daemon runs at most one MPS server at a time, and only one Linux user on the system can have an active server.

- **Server defaults.** The administrator sets defaults through the daemon, with `set_default_active_thread_percentage` for compute and `set_default_device_pinned_mem_limit` for memory. They apply to every client.
- **Per-client limits.** A client narrows its own limits through its environment with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`. These variables can only lower the server's defaults, never raise them.
- **Static SM partitions.** Starting the daemon with `-S` turns on static partitioning, which [Static SM Partitions](#static-sm-partitions) describes.

This works for one user running a fixed set of processes. It fits a cluster of independently scheduled models poorly:

- **Per-model sizes depend on each pod's environment.** Every model with its own size needs the right variables in its own pod, and the server has no record of what each pod is supposed to get.
- **Kubernetes support is equal-sized.** The NVIDIA device plugin's MPS mode, which is experimental, advertises a GPU as a number of equal replicas, each with the same fraction of memory and compute. A model that needs 20% of a GPU and a model that needs 50% can't share it at their own sizes.

## MPS v3: Limits Live on the Server

The v3 interface is selected with `nvidia-cuda-mps-control -p 3` or `CUDA_MPS_PROTOCOL_VERSION=3`. The Kubernetes GPU fractions stack requires NVIDIA driver r615 or newer, with CUDA 13.4. v3 reorganizes MPS into three levels:

1. **Control daemon:** one per node in the Kubernetes stack.
2. **Named servers:** several can run at once. Each server is limited to a list of GPUs through its allowed devices, and the Kubernetes stack starts one server per GPU.
3. **Namespaces:** each server holds namespaces, and each namespace carries its own active thread percentage, pinned memory limit, and client priority.

A client chooses its namespace through its pipe directory, `<pipe>/<server>/<namespace>`. A pod is given only its own namespace's directory, so it can connect to no other namespace, and its limits are not set by anything in its own environment. A client can still lower a namespace's limits for itself, but it cannot raise them.

v3 also adds:

- **cgroup memory limits.** A hard memory limit per container and per GPU, set with `nvidia-smi memory-limits` and enforced through the cgroup v2 `dmem` controller (Linux 6.14 and newer) or the driver's fallback backend. This is the only kind of memory cap that `cudaMemGetInfo` reports as the GPU's total memory, and setting it requires root. It needs CUDA 13.4 or newer and does not work on MIG-enabled GPUs.
- **A configuration file.** The daemon can apply servers, namespaces, and limits from a TOML file at startup with `-a <file>`.
- **Client priority.** Each namespace can run its clients at normal or below-normal priority.

## Static SM Partitions

Both interfaces can give a client an exact set of SMs instead of a percentage ceiling. Static partitions are supported on Ampere and newer GPUs, and on Hopper and newer they are allocated in chunks of 8 SMs. They are the only spatial SM isolation MPS offers, and they change how faults spread: a memory fault stops only the clients in the faulting partition. [What Happens When a Model Faults](how-gpu-sharing-works.md#what-happens-when-a-model-faults) shows what partitions contain and what they don't.

You create a partition through the control daemon, then point a client at it:

- **v2:** start the daemon with `-S`, then send it `sm_partition add <GPU UUID> <chunks>`.
- **v3:** run `sm-partition create`. A partition belongs to a server and a GPU, not to a namespace, so clients of one namespace can run on different partitions.

```bash
nvidia-cuda-mps-control -p 3 sm-partition create model-a --server=gpu0 --device=0 --chunks=11
```

Either command prints the partition's identifier as a `CUDA_MPS_SM_PARTITION=<GPU UUID>/<partition ID>` setting. A client started with that variable runs only on the partition's SMs. Memory bandwidth and the L2 cache are still shared.

## Side by Side

| | MPS v2 (legacy interface) | MPS v3 |
|---|---|---|
| Servers | One active server per daemon, for one Linux user | Several named servers in one daemon, each limited to a list of GPUs |
| Where a client's limits come from | Server defaults, lowered by variables in the client's environment | The client's namespace on the server; the client can lower them but not raise them |
| Compute limit | Active thread percentage, or a static SM partition when the daemon runs with `-S` | Active thread percentage, or a static SM partition |
| Memory faults | Shared by every client on a GPU, or contained to a static SM partition | Shared by every client on a GPU, across namespaces, or contained to a static SM partition |
| Memory limit | Pinned memory limit, one value for all of a client's GPUs | Pinned memory limit per namespace, or a cgroup limit per container and per GPU |
| Different sizes on one GPU | Only through each client's environment | One namespace per model, each with its own limits |
| Kubernetes integration | NVIDIA device plugin MPS mode: equal replicas, experimental | KAI-Scheduler GPU fractions: a size per pod, from pull requests not yet merged upstream |
| Driver | Any MPS-capable driver | r615 or newer for the Kubernetes GPU fractions stack |

## Why GPU Fractions Needs v3

GPU fractions gives every model its own share of a GPU. When KAI-Scheduler places a pod, the node agent creates a namespace for that pod on the GPU's MPS server, sets its limits there, and gives the pod only that namespace's pipe directory. Three properties of v3 make this work:

- **Every model can have a different size.** Each namespace has its own limits, so a model at 20% and a model at 50% share one GPU at the sizes they need.
- **The limits are outside the pod's control.** They are set on the server, not in the pod's environment, so the model can't raise them.
- **Each GPU has its own server.** A problem with one GPU's server affects only the pods on that GPU.

## Sharp Edges in v3

MPS v3 enforces limits only for clients that actually connect to their namespace, and some failures are silent. Run with these in mind, and verify the limits on every new deployment:

- **A client that misses its namespace runs unlimited.** In the default compute mode, a client whose pipe directory is missing or misspelled runs as an ordinary CUDA process with the whole GPU, and nothing reports an error.
- **One bad per-GPU memory entry removes every memory cap.** If any entry in a client's `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` names a GPU the client can't see, the client drops every pinned memory limit, including its namespace's. Its compute limit still applies.
- **A root container can raise its own limits.** The GPU fractions DGDs run workers as non-root for this reason.
- **The reported total memory is the whole GPU.** Under a pinned memory limit, `cudaMemGetInfo` reports the limit as free memory but the whole GPU as total. Set a framework's memory fraction from the cap explicitly. For vLLM, set `--gpu-memory-utilization` to the cap divided by the GPU's total memory, or lower.
- **A server can disappear quietly.** A server whose allowed devices name a GPU that isn't on the node is accepted, then exits, and its clients run without MPS. Check `server list` and the server log after every start.

[Per-GPU Compute and Memory Limits with MPS](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/mps-per-gpu-limits.md) walks through MPS v3 servers, namespaces, and cgroup memory limits by hand, with the behavior measured on an RTX PRO 6000 Blackwell and the fixes for each of these problems.

## Next Steps

- [Which Models to Share a GPU](which-models-to-share.md)
- [Example MPS Architecture](example-mps-architecture.md)
- [Choosing a GPU Sharing Approach](choosing-a-gpu-sharing-approach.md)
