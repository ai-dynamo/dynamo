---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FluxCD
---

This section describes how to use FluxCD for GitOps-based deployment of Dynamo inference graphs. GitOps enables you to manage your Dynamo deployments declaratively using Git as the source of truth. We'll use the [aggregated vLLM example](../knowledge-base/modular-components/backends/vllm/overview.md) to demonstrate the workflow.

## Prerequisites

- A Kubernetes cluster with [Dynamo Kubernetes Platform](../../kubernetes/installation/install-dynamo.md) installed
- [FluxCD](https://fluxcd.io/flux/installation/) installed in your cluster
- A Git repository to store your deployment configurations

## Workflow Overview

The GitOps workflow for Dynamo deployments consists of three main steps:

1. Build and push the Dynamo Operator
2. Create and commit a DynamoGraphDeployment custom resource for initial deployment
3. Update the graph by building a new version and updating the CR for subsequent updates

## Step 1: Build and Push Dynamo Operator

First, follow to [See Install Dynamo Kubernetes Platform](../../kubernetes/installation/install-dynamo.md).

## Step 2: Create Initial Deployment

Create a new file in your Git repository (e.g., `deployments/llm-agg.yaml`) with the following content:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: llm-agg
spec:
  components:
  - name: Frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          env:
          - name: SPECIFIC_ENV_VAR
            value: some_specific_value
  - name: Processor
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          env:
          - name: SPECIFIC_ENV_VAR
            value: some_specific_value
  - name: worker
    replicas: 1
    podTemplate:
      spec:
        # Add PVC for model storage
        volumes:
        - name: vllm-model-storage
          persistentVolumeClaim:
            claimName: vllm-model-storage
        containers:
        - name: main
          env:
          - name: SPECIFIC_ENV_VAR
            value: some_specific_value
          volumeMounts:
          - name: vllm-model-storage
            mountPath: /models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-model-storage
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
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