---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Kubernetes Deployment
subtitle: Use Dynamo's Kubernetes-native path when you are ready to deploy on a GPU cluster.
---

Use the Kubernetes guides when you are ready to move beyond a local Dynamo
process and deploy on a GPU cluster. Dynamo's Kubernetes path is native to the
platform: inference graphs are expressed as Dynamo CRDs, reconciled by the
Dynamo operator, installed with Helm, and integrated with Kubernetes service
discovery, Gateway API Inference Extension, scheduling, observability, and
model-loading workflows.

This does not make Kubernetes the only way to use Dynamo. Local containers,
PyPI installs, and standalone components remain the right path for evaluation,
development, and incremental adoption.

Start with the [Kubernetes Quickstart](../kubernetes/README.md) to run one model end to end. Then use the rest of the Kubernetes Deployment section based on what you need next:

| Goal | Guide |
|---|---|
| Install the operator and prerequisites | [Installation Guide](../kubernetes/installation-guide.md) |
| Deploy and manage models | [Deployment Overview](../kubernetes/model-deployment-guide.md) |
| Load models faster across pods | [Model Caching](../kubernetes/model-caching.md) and [ModelExpress](../kubernetes/modelexpress.md) |
| Operate a cluster deployment | [Autoscaling](../kubernetes/autoscaling.md), [Rolling Update](../kubernetes/rolling-update.md), [Disagg Communication](../kubernetes/disagg-communication-guide.md), and [Observability Metrics](../kubernetes/observability/metrics.md) |
| Scale disaggregated serving | [Multinode Deployments](../kubernetes/deployment/multinode-deployment.md), [Grove](../kubernetes/grove.md), and [Topology Aware Scheduling](../kubernetes/topology-aware-scheduling.md) |
| Integrate with Kubernetes serving APIs | [Gateway API Inference Extension (GAIE)](../kubernetes/inference-gateway.md) and [LWS](../kubernetes/lws.md) |

If you are still evaluating Dynamo locally, start with the [Quickstart](quickstart.mdx) and [Local Installation](local-installation.md) first.
