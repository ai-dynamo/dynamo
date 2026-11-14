---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Choosing a GPU Sharing Approach
subtitle: Deciding between dedicated GPUs, MIG, MPS GPU fractions, and time-slicing, and explaining the decision
---

This page helps you decide whether your project should use MPS GPU fractions, and gives you what you need to explain that decision. It assumes the background in [How GPU Sharing Works](how-gpu-sharing-works.md) and [Which Models to Share a GPU](which-models-to-share.md).

## Four Questions

![A flowchart of four questions. One: does one model have enough traffic to keep a whole GPU busy? If yes, use dedicated GPUs and scale whole-GPU replicas behind the Dynamo router. Two: are the models copies of one model, or LoRA adapters of one base model? If yes, use one engine per base model, because batching and multi-LoRA beat packing. Three: do tenants need hard isolation of faults, memory bandwidth, or security? If yes, use MIG, with fixed sizes and resizing by draining the GPU. Four: do the models need compute caps and to run concurrently, with sizes chosen per model? If yes, use MPS v3 GPU fractions, for distinct, lightly loaded models. If no, time-slicing or HAMi is enough, for dev, test, or bursty and latency-tolerant workloads.](../../../assets/img/gpu-sharing-decision.svg)

Ask the questions in order. The first two rule out sharing for most high-traffic production models. Sharing pays off for a fleet of distinct models that each leave a GPU mostly idle.

## Tradeoffs

| | Dedicated GPUs | MIG | MPS GPU fractions | KAI + HAMi | Time-slicing |
|---|---|---|---|---|---|
| GPUs for N lightly loaded models | N | Fewer, with sizes rounded up to profiles | Fewest; each model gets the share it needs | Fewer, but models take turns | Fewer, but with no limits |
| Throughput kept per shared GPU | All of it | Each instance's own slice | 63.5% to 87.9% of a dedicated GPU, in the A100 benchmark | 44.5% to 72.8%, in the same benchmark | Not limited or measured here |
| Latency interference from neighbors | None | None | Memory bandwidth, L2 cache, and power are shared | Shared; models also wait for their turn | Shared, with no caps |
| Fault isolation | Full | Per instance | None within a GPU; static SM partitions can contain memory faults, but not traps | Per process | Per process |
| Sizing | Whole GPUs | Fixed profiles, up to 7 per GPU | Any share and memory size, per pod | Any memory size | None |
| Resizing one model | Reschedule | Drain and repartition the GPU | Restart that model's pod | Restart that model's pod | Not applicable |
| Maturity on Kubernetes | Standard | Standard, through the GPU Operator | KAI-Scheduler and node agents built from pull requests; needs driver r615+, NRI, and GPU Operator v26.7.1+ | Released components | Standard, through the device plugin |

The throughput rows come from the [GPU Sharing](overview.mdx) benchmark: Qwen3-4B on A100 40GB GPUs, two models per GPU, with 2 to 32 requests resident on each GPU. Your numbers will differ with the model, GPU, and traffic, but the ordering is the point. MPS keeps more of the GPU's throughput than time-slicing, because the two models run at the same time instead of taking turns.

## Making the Case

When you propose MPS GPU fractions, or argue against it, frame it around the decision: how many GPUs your models need, and what the shared GPUs cost you in latency and isolation.

### When the Answer Is Yes

MPS GPU fractions fits when all of these hold:

- **You serve several distinct models, and most of them are lightly loaded.** Each would leave a dedicated GPU mostly idle, and their weights differ, so no single engine can batch them.
- **Each model's latency targets leave room.** The models meet their TTFT and ITL targets at a fraction of a GPU, measured with a busy neighbor.
- **The models trust each other.** They come from the same team or platform, so a shared fault domain on each GPU is acceptable. A fault in one model can force every model on its GPU to restart.
- **You can run the required stack.** Driver r615 or newer, a container runtime with NRI enabled, and KAI-Scheduler and the node agents built from the pinned pull requests.

The argument, in your own numbers:

> We serve N distinct models. Each uses a small part of a GPU, so one GPU per model leaves most of N GPUs idle. GPU fractions packs them onto M GPUs, each model capped at the share of compute and memory it needs, and each still meets its latency targets in our measurements. Unlike MIG, each model gets exactly its share, so fewer GPUs are needed, and one model can be resized without disturbing the others. In the published A100 benchmark, sixteen models ran on eight GPUs instead of sixteen, at 22.9% lower cost per output token than one model per GPU at one request per model.

Back each claim with a measurement from your own models: the share each needs to meet its targets, measured next to a busy neighbor, and the GPU count before and after.

### When the Answer Is No

MPS GPU fractions is the wrong choice when any of these hold, and the reason is worth stating plainly:

- **The model is busy.** One model with enough traffic to fill a GPU loses throughput when it shares. In the A100 benchmark, the cost advantage was gone at 32 concurrent requests per model.
- **The models are replicas.** Replicas, and LoRA adapters of one base model, belong in one engine, which batches them and reads the weights once per step.
- **A neighbor's crash or memory traffic is unacceptable.** MPS shares memory bandwidth within a GPU, and GPU fractions shares the fault domain too. Static SM partitions contain memory faults, but a trap still resets the whole GPU. If one tenant must not affect another, use MIG or separate GPUs.
- **The team can't run pre-release scheduler components.** The GPU fractions stack depends on pull requests not yet merged upstream. Until it is released, MIG and dedicated GPUs are the conservative choice.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| A fault in one model affects every model on its GPU | Put replicas of critical models on different GPUs or nodes; keep the most critical models on dedicated GPUs. Static SM partitions contain memory faults, as [What Happens When a Model Faults](how-gpu-sharing-works.md#what-happens-when-a-model-faults) shows |
| A busy neighbor slows a model through shared memory bandwidth, L2 cache, or power | Measure each share with the rest of the GPU loaded; don't pair two latency-critical models; leave headroom |
| A pod that misses its MPS namespace runs with the whole GPU and no error | Verify every deployment's limits, as in the [smoke test](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/deploy-kai-gpu-fractions.md#verify-the-caps-with-the-smoke-test); run workers as non-root |
| An engine sizes its memory from the GPU's total and fails or overruns its cap | Set the engine's memory fraction from the cap explicitly, such as vLLM's `--gpu-memory-utilization` |
| Updating a DGD in place hangs when every share on a node is allocated | Delete and re-apply the DGD instead of editing it, as described in [Known Issues](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/README.md#known-issues) |
| The scheduler components are pre-release | Pin the commits, validate in staging, and keep a dedicated-GPU or MIG fallback |

## Before You Adopt

Answer these with measurements from your own models before you commit:

1. Which of your models are distinct, and which are replicas or LoRA adapters of a shared base?
2. For each distinct model, what traffic does it see, and what are its TTFT and ITL targets?
3. What share of the SMs and how much memory does each model need to meet its targets, measured with the rest of the GPU loaded?
4. How many GPUs do those shares pack into, compared with one GPU per model, and compared with MIG profiles?
5. Which models are critical enough to keep on dedicated GPUs, or to spread across GPUs?
6. Can your clusters run driver r615 or newer, NRI, and the pinned KAI-Scheduler and node agent builds?

## Further Reading

- [GPU Sharing](overview.mdx): the A100 benchmark behind the throughput and cost numbers.
- [Example MPS Architecture](example-mps-architecture.md): a two-node cluster with MPS and with MIG.
- [`examples/gpu-sharing`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing): manifests, scripts, and deployment guides for every configuration.
