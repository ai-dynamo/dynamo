---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Model Deployment
subtitle: Deploy a model on a GPU cluster with Dynamo's two deployment CRDs — DGD for manual control, DGDR for intent-driven auto deployment.
---

Deploying a model on Dynamo means describing an inference graph — a Frontend plus
one or more workers — as a Kubernetes Custom Resource, then letting the Dynamo
operator reconcile it into running pods, services, and scheduling resources.
There are **two CRDs** you can start from, and the whole section is organized
around choosing between them.

## Two ways to deploy: DGD and DGDR

| | **DynamoGraphDeployment (DGD)** | **DynamoGraphDeploymentRequest (DGDR)** |
|---|---|---|
| **Path** | Manual — you author the spec | Automatic — you describe intent |
| **You provide** | The full spec: components, parallelism, replicas, resource limits | Model, backend, workload, and optional SLA targets |
| **What happens** | The operator reconciles your spec into pods directly | The profiler sizes the deployment and **generates a DGD** for you |
| **Best for** | Known-good configs, tuned recipes, full control | New model/hardware combinations, SLA-driven sizing |

### DGD is the primary path

A **[DynamoGraphDeployment (DGD)](../kubernetes/dgd-guide.md)** is the canonical
resource that serves traffic. You write the spec, `kubectl apply` it, and the
operator runs it. Because you control every field, a DGD works for **any** model
and backend, and it is the object that actually persists and serves. Even when
you use DGDR, what it produces is a DGD. Start here:
**[Deploy with DGD](../kubernetes/dgd-guide.md)**.

### DGDR automates sizing — when your model is supported

A **[DynamoGraphDeploymentRequest (DGDR)](../kubernetes/dgdr-guide.md)** is the
deploy-by-intent path. Instead of hand-authoring parallelism and replica counts,
you describe what you want to run and Dynamo's profiler analyzes your GPUs,
selects a configuration, and generates a DGD.

DGDR is convenient but **does not yet cover every model, hardware, and backend
combination**. Its fast "rapid" profiling relies on the AIConfigurator support
matrix; for combinations outside that matrix it falls back to a naive
memory-fit configuration that may not be optimal. For this reason **DGD remains
the primary path**, and even in the DGDR flow the generated DGD is something you
can inspect and edit before it serves traffic. See
**[Auto Deploy with DGDR](../kubernetes/dgdr-guide.md)**.

## Aggregated or disaggregated — supported by both

Independent of which CRD you author, you choose how prefill and decode are
placed. **Both DGD and DGDR support both options.**

- **Aggregated** — one worker runs both the prefill (prompt) and decode
  (generation) phases. The simplest deployment; a good default for smaller
  models and uniform traffic.
- **Disaggregated** — separate prefill and decode worker pools, each sized and
  scaled independently, with the KV cache transferred between them over the
  network. Use it when prefill and decode have different bottlenecks — long
  prompts saturating prefill, or high concurrency saturating decode. It needs an
  RDMA-capable fabric for the KV transfer to perform well.

See **[Disaggregated Serving](../features/disaggregated-serving/README.md)** for
the architecture and the disaggregation-specific configuration.

## Where to go next

Run one model end to end with the [Kubernetes Quickstart](../kubernetes/README.md)
first if you are new here. Then pick your path:

| Goal | Guide |
|---|---|
| Author a deployment by hand (start here) | [Deploy with DGD](../kubernetes/dgd-guide.md) |
| Let Dynamo size and generate the deployment | [Auto Deploy with DGDR](../kubernetes/dgdr-guide.md) |
| Split prefill and decode for performance | [Disaggregated Serving](../features/disaggregated-serving/README.md) |
| Auto-pick TP/PP/DP for a latency target | [Size with AIConfigurator](../kubernetes/dgd-aiconfigurator.md) |
| Scale a model across multiple nodes | [Multinode Deployments](../kubernetes/deployment/multinode-deployment.md) |
| Understand the full resource model (DGD, DCD, DGDR, recipes) | [Deployment Overview](../kubernetes/model-deployment-guide.md) |
| Load models faster across pods | [Model Caching](../kubernetes/model-caching.md) |

If you are still evaluating Dynamo locally, start with the [Quickstart](quickstart.mdx)
and [Local Installation](local-installation.md) first.
