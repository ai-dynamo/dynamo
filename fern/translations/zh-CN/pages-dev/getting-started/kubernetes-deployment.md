---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Kubernetes 部署
subtitle: 当你准备好在 GPU 集群上部署时，使用 Dynamo 的 Kubernetes 原生路径。
---

当你准备从本地 Dynamo 进程过渡到 GPU 集群部署时，请使用 Kubernetes 指南。
Dynamo 的 Kubernetes 路径是平台原生的：推理图以 Dynamo CRD（Custom
Resource Definition，自定义资源定义）形式表达，由 Dynamo Operator 协调，
通过 Helm 安装，并与 Kubernetes 服务发现、Gateway API Inference
Extension（GAIE，网关 API 推理扩展）、调度、可观测性和模型加载工作流集成。

这并不意味着 Kubernetes 是使用 Dynamo 的唯一方式。
本地容器、PyPI 安装和独立组件仍然是评估、开发和增量采用的正确路径。

首先从 [Kubernetes 快速入门](../../../../../docs/kubernetes/README.md)开始，端到端运行一个模型。
然后根据你接下来的需求使用 Kubernetes 部署部分的其他内容：

| 目标 | 指南 |
|---|---|
| 安装 Operator 和前置依赖 | [安装指南](../../../../../docs/kubernetes/installation-guide.md) |
| 部署和管理模型 | [部署概述](../../../../../docs/kubernetes/model-deployment-guide.md) |
| 跨 Pod 更快加载模型 | [模型缓存](../../../../../docs/kubernetes/model-caching.md) 和 [ModelExpress](../../../../../docs/kubernetes/modelexpress.md) |
| 运维集群部署 | [自动扩缩容](../../../../../docs/kubernetes/autoscaling.md)、[滚动更新](../../../../../docs/kubernetes/rolling-update.md)、[解耦通信](../../../../../docs/kubernetes/disagg-communication-guide.md)和[可观测性指标](../../../../../docs/kubernetes/observability/metrics.md) |
| 以 Kubernetes 原生方式路由流量 | [Gateway API Inference Extension (GAIE)](../../../../../docs/kubernetes/gateway-api/README.mdx) |
| 扩展解耦服务 | [多节点部署](../../../../../docs/kubernetes/deployment/multinode-deployment.md)、[Grove](../../../../../docs/kubernetes/grove.md) 和[拓扑感知调度](../../../../../docs/kubernetes/topology-aware-scheduling.md) |
| 调度器支持 | [Grove](../../../../../docs/kubernetes/grove.md) 和 [LWS](../../../../../docs/kubernetes/lws.md) |

如果你仍在本地评估 Dynamo，请先从[快速入门](../../../../../docs/getting-started/quickstart.mdx)和[本地安装](../../../../../docs/getting-started/local-installation.md)开始。
