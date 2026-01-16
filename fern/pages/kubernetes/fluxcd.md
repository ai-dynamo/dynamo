---
title: "GitOps Deployment with FluxCD"
---

{/*
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/}

This section describes how to use FluxCD for GitOps-based deployment of Dynamo inference graphs. GitOps enables you to manage your Dynamo deployments declaratively using Git as the source of truth. We'll use the [aggregated vLLM example](../backends/vllm/README.md) to demonstrate the workflow.

## Prerequisites

- A Kubernetes cluster with [Dynamo Kubernetes Platform](installation-guide.md) installed
- [FluxCD](https://fluxcd.io/flux/installation/) installed in your cluster
- A Git repository to store your deployment configurations

## Workflow Overview

The GitOps workflow for Dynamo deployments consists of three main steps:

1. Build and push the Dynamo Operator
2. Create and commit a DynamoGraphDeployment custom resource for initial deployment
3. Update the graph by building a new version and updating the CR for subsequent updates

## Step 1: Build and Push Dynamo Operator

First, follow to [See Install Dynamo Kubernetes Platform](installation-guide.md).

## Step 2: Create Initial Deployment

Create a new file in your Git repository (e.g., `deployments/llm-agg.yaml`) with the following content:

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
      # Add PVC for model storage
      volumeMounts:
        - name: vllm-model-storage
          mountPoint: /models
```

Commit and push this file to your Git repository. FluxCD will detect the new CR and create the initial Dynamo deployment in your cluster.

## Step 3: Update Existing Deployment

To update your pipeline, just update the associated DynamoGraphDeployment CRD

The Dynamo operator will automatically reconcile it.

## Monitoring the Deployment

You can monitor the deployment status using:

```bash

export NAMESPACE=<namespace-with-the-dynamo-operator>

# Check the DynamoGraphDeployment status
kubectl get dynamographdeployment llm-agg -n $NAMESPACE
```