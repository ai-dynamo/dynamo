---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Kubernetes Deployment
subtitle: Follow the Kubernetes guides when you are ready to deploy Dynamo on a GPU cluster.
---

Use the Kubernetes guides when you are ready to move beyond a local Dynamo process and deploy on a GPU cluster.

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
