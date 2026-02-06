<!--
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
-->

# Installation Guide for Dynamo Kubernetes Platform

Deploy and manage Dynamo inference graphs on Kubernetes with automated orchestration and scaling, using the Dynamo Kubernetes Platform.

## Before You Start

Determine your cluster environment:

**Shared/Multi-Tenant Cluster** (K8s cluster with existing Dynamo artifacts):
- CRDs already installed cluster-wide - skip CRD installation step
- A cluster-wide Dynamo operator is likely already running
- **Do NOT install another operator** - use the existing cluster-wide operator
- Only install a namespace-restricted operator if you specifically need to prevent the cluster-wide operator from managing your namespace (e.g., testing operator features you're developing)

**Dedicated Cluster** (full cluster admin access):
- You install CRDs yourself
- Can use cluster-wide operator (default)

**Local Development** (Minikube, testing):
- See [Minikube Setup](deployment/minikube.md) first, then follow installation steps below

To check if CRDs already exist:
```bash
kubectl get crd | grep dynamo
# If you see dynamographdeployments, dynamocomponentdeployments, etc., CRDs are already installed
```

To check if a cluster-wide operator already exists:
```bash
# Check for cluster-wide operator and show its namespace
kubectl get clusterrolebinding -o json | \
  jq -r '.items[] | select(.metadata.name | contains("dynamo-operator-manager")) |
  "Cluster-wide operator found in namespace: \(.subjects[0].namespace)"'

# If a cluster-wide operator exists: Do NOT install another operator
# Only install namespace-restricted mode if you specifically need namespace isolation
```

## Installation Paths

Platform is installed using Dynamo Kubernetes Platform [helm chart](../../deploy/helm/charts/platform/README.md).

**Path A: Pre-built Artifacts**
- Use case: Production deployment, shared or dedicated clusters
- Source: NGC published Helm charts
- Time: ~10 minutes
- Jump to: [Path A](#path-a-production-install)

**Path B: Custom Build from Source**
- Use case: Contributing to Dynamo, using latest features from main branch, customization
- Requirements: Docker build environment
- Time: ~30 minutes
- Jump to: [Path B](#path-b-custom-build-from-source)

All helm install commands could be overridden by either setting the values.yaml file or by passing in your own values.yaml:

```bash
helm install ...
  -f your-values.yaml
```

and/or setting values as flags to the helm install command, as follows:

```bash
helm install ...
  --set "your-value=your-value"
```

## Prerequisites

Before installing the Dynamo Kubernetes Platform, ensure you have the following tools and access:

### Required Tools

| Tool | Minimum Version | Description | Installation |
|------|-----------------|-------------|--------------|
| **kubectl** | v1.24+ | Kubernetes command-line tool | [Install kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) |
| **Helm** | v3.0+ | Kubernetes package manager | [Install Helm](https://helm.sh/docs/intro/install/) |
| **Docker** | Latest | Container runtime (Path B only) | [Install Docker](https://docs.docker.com/get-docker/) |

### Cluster and Access Requirements

- **Kubernetes cluster v1.24+** with admin or namespace-scoped access
- **Cluster type determined** (shared vs dedicated) — see [Before You Start](#before-you-start)
- **CRD status checked** if on a shared cluster
- **NGC credentials** (optional) — required only if pulling NVIDIA images from NGC

### Verify Installation

Run the following to confirm your tools are correctly installed:

```bash
# Verify tools and versions
kubectl version --client  # Should show v1.24+
helm version              # Should show v3.0+
docker version            # Required for Path B only

# Set your release version
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases
```

### Pre-Deployment Checks

Before proceeding, run the pre-deployment check script to verify your cluster meets all requirements:

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

This script validates kubectl connectivity, default StorageClass configuration, and GPU node availability. See [Pre-Deployment Checks](../../deploy/pre-deployment/README.md) for details.

> **No cluster?** See [Minikube Setup](deployment/minikube.md) for local development.

**Estimated installation time:** 5-30 minutes depending on path

## Path A: Production Install

Install from [NGC published artifacts](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts).

```bash
# 1. Set environment
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases

# 2. Install CRDs (skip if on shared cluster where CRDs already exist)
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# 3. Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
```

**For Shared/Multi-Tenant Clusters:**

If your cluster has namespace-restricted Dynamo operators, you MUST add namespace restriction to your installation:

```bash
# Add this flag to the helm install command above
--set dynamo-operator.namespaceRestriction.enabled=true
```

Note: Use the full path `dynamo-operator.namespaceRestriction.enabled=true` (not just `namespaceRestriction.enabled=true`).

If you see this validation error, you need namespace restriction:
```
VALIDATION ERROR: Cannot install cluster-wide Dynamo operator.
Found existing namespace-restricted Dynamo operators in namespaces: ...
```

> [!TIP]
> For multinode deployments, you need to install multinode orchestration components:
>
> **Option 1 (Recommended): Grove + KAI Scheduler**
> - Grove and KAI Scheduler can be installed manually or through the dynamo-platform helm install command.
> - When using the dynamo-platform helm install command, Grove and KAI Scheduler are NOT installed by default. You can enable their installation by setting the following flags:
>
> ```bash
> --set "grove.enabled=true"
> --set "kai-scheduler.enabled=true"
> ```
>
> **Option 2: LeaderWorkerSet (LWS) + Volcano**
> - If using LWS for multinode deployments, you must also install Volcano (required dependency):
>   - [LWS Installation](https://github.com/kubernetes-sigs/lws#installation)
>   - [Volcano Installation](https://volcano.sh/en/docs/installation/) (required for gang scheduling with LWS)
> - These must be installed manually before deploying multinode workloads with LWS.
>
> See the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for details on orchestrator selection.

> [!TIP]
> By default, Model Express Server is not used.
> If you wish to use an existing Model Express Server, you can set the modelExpressURL to the existing server's URL in the helm install command:

```bash
--set "dynamo-operator.modelExpressURL=http://model-express-server.model-express.svc.cluster.local:8080"
```

> [!TIP]
> By default, Dynamo Operator is installed cluster-wide and will monitor all namespaces.
> If you wish to restrict the operator to monitor only a specific namespace (the helm release namespace by default), you can set the namespaceRestriction.enabled to true.
> You can also change the restricted namespace by setting the targetNamespace property.

```bash
--set "dynamo-operator.namespaceRestriction.enabled=true"
--set "dynamo-operator.namespaceRestriction.targetNamespace=dynamo-namespace" # optional
```

→ [Verify Installation](#verify-installation)

## Path B: Custom Build from Source

Build and deploy from source for customization, contributing to Dynamo, or using the latest features from the main branch.

Note: This gives you access to the latest unreleased features and fixes on the main branch.

```bash
# 1. Set environment
export NAMESPACE=dynamo-system
export DOCKER_SERVER=nvcr.io/nvidia/ai-dynamo/  # or your registry
export DOCKER_USERNAME='$oauthtoken'
export DOCKER_PASSWORD=<YOUR_NGC_CLI_API_KEY>
export IMAGE_TAG=${RELEASE_VERSION}

# 2. Build operator
cd deploy/operator

# 2.1 Alternative 1 : Build and push the operator image for multiple platforms
docker buildx create --name multiplatform --driver docker-container --bootstrap
docker buildx use multiplatform
docker buildx build --platform linux/amd64,linux/arm64 -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG --push .

# 2.2 Alternative 2 : Build and push the operator image for a single platform
docker build -t $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG . && docker push $DOCKER_SERVER/dynamo-operator:$IMAGE_TAG

cd -

# 3. Create namespace and secrets to be able to pull the operator image (only needed if you pushed the operator image to a private registry)
kubectl create namespace ${NAMESPACE}
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=${DOCKER_SERVER} \
  --docker-username=${DOCKER_USERNAME} \
  --docker-password=${DOCKER_PASSWORD} \
  --namespace=${NAMESPACE}

cd deploy/helm/charts

# 4. Install CRDs
helm upgrade --install dynamo-crds ./crds/ --namespace default

# 5. Install Platform
helm dep build ./platform/

# To install cluster-wide instead, set NS_RESTRICT_FLAGS="" (empty) or omit that line entirely.

NS_RESTRICT_FLAGS="--set dynamo-operator.namespaceRestriction.enabled=true"
helm install dynamo-platform ./platform/ \
  --namespace "${NAMESPACE}" \
  --set "dynamo-operator.controllerManager.manager.image.repository=${DOCKER_SERVER}/dynamo-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=${IMAGE_TAG}" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret" \
  ${NS_RESTRICT_FLAGS}

```

→ [Verify Installation](#verify-installation)

## Verify Installation

```bash
# Check CRDs
kubectl get crd | grep dynamo

# Check operator and platform pods
kubectl get pods -n ${NAMESPACE}
# Expected: dynamo-operator-* and etcd-* and nats-* pods Running
```

## Next Steps

1. **Deploy Model/Workflow**
   ```bash
   # Example: Deploy a vLLM workflow with Qwen3-0.6B using aggregated serving
   kubectl apply -f examples/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

   # Port forward and test
   kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}
   curl http://localhost:8000/v1/models
   ```

2. **Explore Backend Guides**
   - [vLLM Deployments](../../examples/backends/vllm/deploy/README.md)
   - [SGLang Deployments](../../examples/backends/sglang/deploy/README.md)
   - [TensorRT-LLM Deployments](../../examples/backends/trtllm/deploy/README.md)

3. **Optional:**
   - [Set up Prometheus & Grafana](./observability/metrics.md)
   - [SLA Planner Guide](../components/planner/planner_guide.md) (for SLA-aware scheduling and autoscaling)

## Troubleshooting

**"VALIDATION ERROR: Cannot install cluster-wide Dynamo operator"**

```
VALIDATION ERROR: Cannot install cluster-wide Dynamo operator.
Found existing namespace-restricted Dynamo operators in namespaces: ...
```

Cause: Attempting cluster-wide install on a shared cluster with existing namespace-restricted operators.

Solution: Add namespace restriction to your installation:
```bash
--set dynamo-operator.namespaceRestriction.enabled=true
```

Note: Use the full path `dynamo-operator.namespaceRestriction.enabled=true` (not just `namespaceRestriction.enabled=true`).

**CRDs already exist**

Cause: Installing CRDs on a cluster where they're already present (common on shared clusters).

Solution: Skip step 2 (CRD installation), proceed directly to platform installation.

To check if CRDs exist:
```bash
kubectl get crd | grep dynamo
```

**Pods not starting?**
```bash
kubectl describe pod <pod-name> -n ${NAMESPACE}
kubectl logs <pod-name> -n ${NAMESPACE}
```

**HuggingFace model access?**
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

**Bitnami etcd "unrecognized" image?**

```bash
ERROR: Original containers have been substituted for unrecognized ones. Deploying this chart with non-standard containers is likely to cause degraded security and performance, broken chart features, and missing environment variables.
```
This error that you might encounter during helm install is due to bitnami changing their docker repository to a [secure one](https://github.com/bitnami/charts/tree/main/bitnami/etcd#%EF%B8%8F-important-notice-upcoming-changes-to-the-bitnami-catalog).

just add the following to the helm install command:
```bash
--set "etcd.image.repository=bitnamilegacy/etcd" --set "etcd.global.security.allowInsecureImages=true"
```

**Clean uninstall?**

To uninstall the platform, you can run the following command:
```
helm uninstall dynamo-platform --namespace ${NAMESPACE}
```

To uninstall the CRDs, follow these steps:

Get all of the dynamo CRDs installed in your cluster:
```bash
kubectl get crd | grep "dynamo.*nvidia.com"
```

You should see something like this:
```
dynamocomponentdeployments.nvidia.com               2025-10-21T14:49:52Z
dynamocomponents.nvidia.com                         2025-10-25T05:16:10Z
dynamographdeploymentrequests.nvidia.com            2025-11-24T05:26:04Z
dynamographdeployments.nvidia.com                   2025-09-04T20:56:40Z
dynamographdeploymentscalingadapters.nvidia.com     2025-12-09T21:05:59Z
dynamomodels.nvidia.com                             2025-11-07T00:19:43Z
```

Delete each CRD one by one:
```bash
kubectl delete crd <crd-name>
```

## Advanced Options

- [Helm Chart Configuration](../../deploy/helm/charts/platform/README.md)
- [Create custom deployments](./deployment/create_deployment.md)
- [Dynamo Operator details](./dynamo_operator.md)
- [Model Express Server details](https://github.com/ai-dynamo/modelexpress)
