---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FluxCD
---

本节介绍如何使用 FluxCD 进行基于 GitOps 的 Dynamo 推理图部署。
GitOps 允许你以声明式方式管理 Dynamo 部署，将 Git 作为事实来源。
我们将使用[聚合式 vLLM 示例](../../../../../docs/backends/vllm/README.md)来演示此工作流。

## 前置依赖

- 已安装 [Dynamo Kubernetes Platform](../../../../../docs/kubernetes/installation-guide.md) 的 Kubernetes 集群
- 集群中已安装 [FluxCD](https://fluxcd.io/flux/installation/)
- 用于存储部署配置的 Git 仓库

## 工作流概述

Dynamo 部署的 GitOps 工作流包含三个主要步骤：

1. 构建并推送 Dynamo Operator
2. 创建并提交 DynamoGraphDeployment 自定义资源以进行初始部署
3. 通过构建新版本并更新 CR（Custom Resource，自定义资源）来更新推理图，以进行后续更新

## 步骤 1：构建并推送 Dynamo Operator

首先，请按照[安装 Dynamo Kubernetes Platform](../../../../../docs/kubernetes/installation-guide.md)
中的步骤操作。

## 步骤 2：创建初始部署

在你的 Git 仓库中创建一个新文件（例如 `deployments/llm-agg.yaml`），内容如下：

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: llm-agg
spec:
  pvcs:
    - name: vllm-model-storage
      size: 100Gi
  services:
    Frontend:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
    Processor:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
    VllmWorker:
      replicas: 1
      envs:
      - name: SPECIFIC_ENV_VAR
        value: some_specific_value
      # 添加用于模型存储的 PVC
      volumeMounts:
        - name: vllm-model-storage
          mountPoint: /models
```

提交并推送此文件到你的 Git 仓库。FluxCD 会检测到新的 CR，并在你的集群中创建初始的 Dynamo 部署

## 步骤 3：更新现有部署

要更新你的流水线，只需更新相关的
DynamoGraphDeployment CRD（Custom Resource Definition，自定义资源定义）即可。

Dynamo Operator 会自动协调更新。

## 监控部署

你可以使用以下命令监控部署状态：

```bash

export NAMESPACE=<namespace-with-the-dynamo-operator>

# 检查 DynamoGraphDeployment 状态
kubectl get dynamographdeployment llm-agg -n $NAMESPACE
```
