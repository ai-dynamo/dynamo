---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Example MPS Architecture
subtitle: A two-node Kubernetes cluster serving many distinct models with MPS GPU fractions, and the same models with MIG
---

This page walks through one example deployment: a Kubernetes cluster with two GPU nodes, serving fifteen distinct models on GPU fractions and one large model on whole GPUs. It shows which components do what, how to think about a GPU as a set of budgets, and what the same placement looks like with MIG instead of MPS.

The model names, shares, and memory sizes are illustrative. Size your own models as described in [Sizing Each Share](which-models-to-share.md#sizing-each-share).

## The Cluster

![A Kubernetes cluster. Cluster services run KAI-Scheduler, which picks a GPU and a share for each worker pod; the Dynamo operator, with one DynamoGraphDeployment per model; Dynamo frontends, one per model on CPU; and the GPU Operator. Two GPU nodes each have four H100 80GB GPUs and run mpsd, an MPS v3 daemon with one server per GPU, and fractiond, which joins each pod to its namespace. On gpu-node-1, GPU 0 holds chat-8b at 0.5 of the SMs and 32 GB, code-7b at 0.3 and 24 GB, and guard-1b at 0.2 and 8 GB. GPU 1 holds tenant-a-8b and tenant-b-8b at 0.4 and 28 GB each, and embed at 0.2 and 6 GB. GPU 2 holds sql-7b at 0.3 and 24 GB, summarize-3b at 0.3 and 16 GB, and rerank at 0.2 and 6 GB, with 0.2 free. GPU 3 is free. On gpu-node-2, GPUs 0 and 1 run the two ranks of chat-70b on whole GPUs without MPS. GPU 2 holds four tenant models at 0.25 and 18 GB each. GPU 3 holds vision-7b at 0.6 and 40 GB and ocr-2b at 0.2 and 10 GB, with 0.2 free.](../../../assets/img/gpu-sharing-two-node-architecture.svg)

Each model is its own DynamoGraphDeployment (DGD): a frontend pod on a CPU node and a worker pod on a GPU. The figure shows only the workers. Each colored block is one worker pod in its own MPS namespace, and its height is the pod's share of the GPU's SMs.

| Component | Where it runs | What it does |
|---|---|---|
| KAI-Scheduler | Cluster services | Places each fractional worker pod on a GPU with room for both its compute share and its memory, and records the placement on the pod |
| Dynamo operator | Cluster services | Turns each DGD into a frontend and a worker |
| Dynamo frontends | CPU nodes | Serve each model's OpenAI-compatible API and route to that model's worker |
| GPU Operator | Every GPU node | Installs the driver (r615 or newer for MPS v3), the device plugin, and the GPU labels KAI-Scheduler reads |
| `mpsd` | Every GPU node, as a DaemonSet | Runs the MPS v3 control daemon, one MPS server per GPU, and creates a namespace for each fractional pod |
| `fractiond` | Every GPU node, as an NRI plugin | Gives each fractional container its namespace's pipe directory and its limits |

KAI-Scheduler and the `kai-gpu-fractioning` node agents (`mpsd` and `fractiond`) are built from pull requests that are not yet merged upstream. [Build the KAI-Scheduler and GPU Fractioning Forks](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/build-gpu-fractioning-forks.md) has the pinned commits and the node requirements.

## From a Share to an MPS Namespace

A worker asks for its share with annotations on its pod template, and runs as a non-root user so it can't raise its own limits. To give a model the same share of compute and memory, request a fraction:

```yaml
- name: worker
  type: worker
  podTemplate:
    metadata:
      labels:
        kai.scheduler/queue: default-queue
      annotations:
        gpu-fraction: "0.5"
        nvidia.com/container.main.gpu-compute.mode: sm-sharing
    spec:
      schedulerName: kai-scheduler
      runtimeClassName: nvidia
      securityContext:
        runAsNonRoot: true
```

To size compute and memory separately, as for `chat-8b` in the figure (0.5 of the SMs, 32 GB), request each one instead of `gpu-fraction`:

```yaml
annotations:
  nvidia.com/container.main.gpu-compute.mode: sm-sharing
  nvidia.com/container.main.gpu-compute.request: "0.5"
  nvidia.com/container.main.gpu-memory.request: "32768Mi"
```

The worker must not also request `nvidia.com/gpu`; KAI-Scheduler rejects a pod that has both. From there:

1. KAI-Scheduler picks a GPU whose free compute share and free memory both fit the request, and records the GPU and the share on the pod.
2. On that node, `mpsd` creates a namespace for the pod on the GPU's MPS server, with an active thread percentage and a pinned memory limit from the request.
3. When the container starts, `fractiond` mounts only that namespace's pipe directory into it and sets its environment.
4. The engine, such as vLLM, starts as an MPS client of its namespace and runs under that namespace's limits.

Workers that ask for whole GPUs, such as `chat-70b` on gpu-node-2, use `nvidia.com/gpu` as usual and run without MPS. Whole-GPU and fractional workers can share a node.

## How to Think About Each GPU

Think of each GPU as two budgets, and pack models into it:

- **Compute:** the SM shares on a GPU add up to at most 1. On gpu-node-1 GPU 0, `chat-8b` at 0.5, `code-7b` at 0.3, and `guard-1b` at 0.2 fill it exactly.
- **Memory:** the memory requests fit in the GPU's 80 GB. On the same GPU they add up to 64 GB.

A model fits on a GPU only when both budgets have room. GPU 2 on gpu-node-1 has 0.2 of its compute and 34 GB of memory free, so it can take another model that needs at most 0.2 of the SMs, such as a second embedding model. GPU 3 is completely free, so it can take a whole-GPU model or another set of fractions.

Some resources are outside both budgets: memory bandwidth, the L2 cache, power, and the fault domain. Every model on a GPU shares them, so leave headroom and spread critical models:

- **Don't put two latency-critical models on one GPU.** If one is busy, its memory traffic slows the other.
- **Spread the replicas of a critical model across GPUs or nodes.** A fatal fault in any model on a GPU affects every model on that GPU. If `code-7b` hits a fatal GPU fault, `chat-8b` and `guard-1b` on the same GPU are affected too, while every other GPU in the cluster keeps serving. [What Happens When a Model Faults](how-gpu-sharing-works.md#what-happens-when-a-model-faults) shows which faults spread, and how static SM partitions contain memory faults.

## The Same Models With MIG

MIG can also put several models on one GPU, but each model gets a fixed instance, and the menu of instance sizes is short. Take the three models on gpu-node-1 GPU 0:

![What each model needs: chat-8b, 50% of the SMs and 32 GB; code-7b, 30% and 24 GB; guard-1b, 20% and 8 GB, adding up to 100% of the SMs and 64 GB of 80 GB. With MPS, each model gets exactly its share, all three fit, and 16 GB of memory is left over. With MIG, each model is rounded up to a fixed profile: chat-8b gets a 4g.40gb instance with 57% of the SMs and code-7b gets a 3g.40gb instance with 43%, which uses the whole GPU, so guard-1b doesn't fit and needs another GPU. MPS leaves memory bandwidth, L2 cache, power and the fault domain shared; MIG walls all of them off.](../../../assets/img/gpu-sharing-mig-vs-mps.svg)

| Model | Needs | MPS share | Smallest MIG instance that fits |
|---|---|---|---|
| `chat-8b` | 50% of the SMs, 32 GB | 0.5, 32 GB | `4g.40gb`: 57% of the SMs, 40 GB |
| `code-7b` | 30% of the SMs, 24 GB | 0.3, 24 GB | `3g.40gb`: 43% of the SMs, 40 GB. A `2g.20gb` has too little memory |
| `guard-1b` | 20% of the SMs, 8 GB | 0.2, 8 GB | `2g.20gb`, but the GPU's 7 SM slices are already allocated |

With MIG, the first two models take the whole GPU, and the third needs another GPU. Across the cluster, rounding every model up to a profile strands capacity on every GPU.

MIG also changes how the cluster is operated:

- **The layout is chosen per GPU, ahead of time.** On Kubernetes, the GPU Operator's MIG manager applies a layout chosen for each node through a node label. Pods then request a profile, such as `nvidia.com/mig-3g.40gb: 1`.
- **Changing the layout drains the GPU.** Instances can be created or destroyed only when nothing runs in them, so resizing one model's instance disrupts every model on that GPU. Under GPU fractions, resizing one model restarts only that model's pod.
- **The walls are real.** Under MIG, `code-7b`'s memory traffic can't slow `chat-8b`, and a fatal fault in `code-7b` affects only `code-7b`. Under GPU fractions, neither is true. Static SM partitions would keep a memory fault in `code-7b` away from its neighbors, but not its memory traffic.

That is the whole tradeoff: MPS fits more distinct models on each GPU and lets you resize them one at a time, while MIG isolates them completely in fixed sizes.

## Next Steps

- [Choosing a GPU Sharing Approach](choosing-a-gpu-sharing-approach.md): how to decide, and how to explain the decision.
- [Deploy the KAI + GPU Fractions Experiment](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/deploy-kai-gpu-fractions.md): a working GPU fractions deployment, step by step.
