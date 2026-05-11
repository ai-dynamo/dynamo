---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Installation Guide
---

This guide walks you through installing everything needed to deploy models with Dynamo on Kubernetes. Follow the steps in order — each builds on the previous one.

## Prerequisites

Before you begin, make sure you have:

- A **Kubernetes cluster (v1.24+)** with GPU-capable nodes. See the cloud provider guides if you need to create one:
  - [Amazon EKS](cloud-providers/eks/eks.md) | [Azure AKS](cloud-providers/aks/aks.md) | [Google GKE](cloud-providers/gke/gke.md)
  - For local development: [Minikube Setup](deployment/minikube.md)
- **kubectl** v1.24+ — [Install kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
- **Helm** v3.0+ — [Install Helm](https://helm.sh/docs/intro/install/)

> [!IMPORTANT]
> **Cloud provider GPU drivers**: The GPU Operator (Step 1) installs GPU drivers for you. When creating your cluster's GPU node pools, **do not enable provider-managed GPU driver installation** (e.g., skip AKS GPU driver install, don't use GKE `--accelerator gpu-driver-version=latest`). If your nodes already have provider-managed drivers, see the GPU Operator step for how to handle this.

Verify your tools:

```bash
kubectl version --client  # Should show v1.24+
helm version              # Should show v3.0+
```

## Overview

Every Dynamo deployment requires two Helm charts: the **GPU Operator** (Step 1) and the **Dynamo Platform** (Step 2). Everything else is optional. Decide what optional components you need before starting so you can install them in Step 3.

| Optional Component | When you need it | Required for |
|-----------|-----------------|--------------|
| Grove + KAI Scheduler | Multinode or disaggregated inference | Multinode deployments (operator errors without Grove or LWS) |
| Network Operator / RDMA | Disaggregated inference in production | Acceptable KV cache transfer performance (TCP fallback has ~200-500x degradation) |
| kube-prometheus-stack | Autoscaling, metrics dashboards, or the Planner | Planner `sla` mode, KEDA/HPA autoscaling |
| Shared storage (model cache) | Large models (>70B) or many replicas | Avoiding per-pod downloads and HuggingFace rate limits |

**Grove + KAI Scheduler** — Grove is the default multinode orchestrator. The operator returns a hard error on multinode deployments if neither Grove nor [LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws#installation) is available. KAI Scheduler is optional but recommended alongside Grove for GPU-aware scheduling. See [Grove](grove.md) for details.

**Network Operator / RDMA** — Without RDMA, disaggregated inference falls back to TCP automatically, but with severe performance degradation (~98s TTFT vs ~200-500ms with RDMA). Required for any production disaggregated deployment. Setup is cloud-provider-specific — see the [Disaggregated Communication Guide](disagg-communication-guide.md) and your cloud provider guide.

**kube-prometheus-stack** — Required for the Planner's `sla` optimization mode (it reads live TTFT/ITL metrics from Prometheus). Also required for KEDA/HPA-based autoscaling. The Planner's `throughput` mode can function without it using internal queue depth signals, but metrics-driven features will not work. See [Metrics](observability/metrics.md) for details.

**Shared storage** — Prevents each pod from downloading model weights independently. Without it, large models (>70B) take hours to download per pod, and many replicas will hit HuggingFace rate limits. Not enforced by the operator — this is an operational concern. See [Model Caching](model-caching.md) for the full walkthrough.

## Step 1: Install the GPU Operator

The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) automates deployment of all NVIDIA software components needed to provision GPUs — drivers, container toolkit, device plugin, and monitoring.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace
  # Uncomment if your nodes already have provider-managed GPU drivers installed:
  # --set driver.enabled=false
```

If your GPU nodes already have provider-managed drivers installed (e.g., you used GKE's `--accelerator gpu-driver-version=latest`), uncomment the `driver.enabled=false` line above so the operator doesn't conflict with the existing drivers.

> [!NOTE]
> Some cloud providers require additional GPU Operator configuration. See your provider guide for details:
> - [AKS GPU Operator setup](cloud-providers/aks/aks.md) — skip AKS-managed GPU driver install on node pools
> - [EKS GPU Operator setup](cloud-providers/eks/eks.md)
> - [GKE GPU Operator setup](cloud-providers/gke/gke.md) — `LD_LIBRARY_PATH` and `ldconfig` init requirements

Verify the GPU Operator is running:

```bash
kubectl get pods -n gpu-operator
# Expected: gpu-operator, nvidia-driver-daemonset, nvidia-device-plugin-daemonset, etc. all Running
```

## Step 2: Install the Dynamo Platform

Set your environment variables:

```bash
export NAMESPACE=dynamo-system
export RELEASE_VERSION=1.0.2  # match a version from https://github.com/ai-dynamo/dynamo/releases
```

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-$RELEASE_VERSION.tgz
helm install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace
  # Note: add \ to --create-namespace above when uncommenting any optional flags below
  #
  # Grove + KAI Scheduler — uncomment if using multinode or disaggregated inference.
  # Option A (install=true): Dynamo installs and manages Grove/KAI as bundled subcharts (dev/testing):
  # --set "global.grove.install=true" \
  # --set "global.kai-scheduler.install=true" \
  # Option B (enabled=true): Grove/KAI are already installed externally (production):
  # --set "global.grove.enabled=true" \
  # --set "global.kai-scheduler.enabled=true" \
  #
  # kube-prometheus-stack — uncomment if Prometheus is installed (required for Planner sla mode and autoscaling):
  # --set "dynamo-operator.dynamo.metrics.prometheusEndpoint=http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
```

> [!TIP]
> All `helm install` commands can be customized with your own values file: `helm install ... -f your-values.yaml`

> [!TIP]
> **Shared/Multi-Tenant Clusters**: If a cluster-wide Dynamo operator is already running, do **not** install another one. Check with:
> ```bash
> kubectl get clusterrolebinding -o json | \
>   jq -r '.items[] | select(.metadata.name | contains("dynamo-operator-manager")) |
>   "Cluster-wide operator found in namespace: \(.subjects[0].namespace)"'
> ```

> [!WARNING]
> **Namespace-restricted mode** (`namespaceRestriction.enabled=true`) is deprecated and will be removed in a future release. Use the default cluster-wide mode for all new deployments.

Verify the Dynamo platform is running:

```bash
# Check CRDs
kubectl get crd | grep dynamo
# Expected: dynamographdeployments, dynamocomponentdeployments, dynamographdeploymentrequests, etc.

# Check operator and platform pods
kubectl get pods -n $NAMESPACE
# Expected: dynamo-operator-*, etcd-*, nats-* pods all Running
```

## Step 3: Install Optional Components

The Dynamo install command above includes commented flags for each optional component. Install the component first, then uncomment the corresponding flag before running `helm install` in Step 2 (or run `helm upgrade --reuse-values` with the flag if you've already installed Dynamo).

### Multinode:

Multinode deployments require either Grove + KAI Scheduler or an alternative orchestrator setup (LeaderWorkerSet + Volcano) to enable gang scheduling for workloads that span multiple nodes. See the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for details on orchestrator selection and configuration.

#### Grove + KAI Scheduler

There are two ways to enable Grove and KAI Scheduler, controlled by which flags you uncomment in the Dynamo install command:

- **`install=true`** — Dynamo installs and manages Grove/KAI as bundled subcharts. Simplest path; recommended for dev/testing.
- **`enabled=true`** — Tells Dynamo that Grove/KAI are already installed and externally managed. Use this when you install Grove/KAI separately (e.g., to manage their lifecycle independently or share them across namespaces). Recommended for production.

For the `enabled=true` path, install Grove and KAI Scheduler separately first. See the [Grove installation guide](https://github.com/NVIDIA/grove/blob/main/docs/installation.md) and [KAI Scheduler deployment guide](https://github.com/NVIDIA/KAI-Scheduler) for instructions.

> [!NOTE]
> **Compatibility matrix:**
>
> | dynamo-platform | kai-scheduler | Grove |
> |-----------------|---------------|-------|
> | 1.0.x           | >= v0.13.0    | >= v0.1.0-alpha.6 |
> | 1.1.x           | >= v0.13.4    | >= v0.1.0-alpha.8 |

#### LWS + Volcano

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) (>= v0.7.0) with [Volcano](https://volcano.sh/en/docs/installation/) for gang scheduling. Both must be installed before deploying multinode workloads.

1. Install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. Install LWS (>= v0.7.0) with Volcano gang scheduling enabled:

```bash
export LWS_VERSION=0.8.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://volcano.sh/en/docs/) for configuration options, and the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for orchestrator selection.

### Network Operator / RDMA

RDMA setup is cloud-provider-specific. See the [Disaggregated Communication Guide](disagg-communication-guide.md) for transport options, UCX configuration, and performance expectations, and your cloud provider guide for setup instructions:

- [AKS — InfiniBand + Network Operator](cloud-providers/aks/rdma-infiniband.md)
- [EKS — EFA device plugin](cloud-providers/eks/eks.md) (also see the [EFA configuration guide](disagg-communication-guide.md#aws-efa-configuration))
- [GKE — GPUDirect-TCPXO](cloud-providers/gke/gke.md)

### kube-prometheus-stack

Install Prometheus before running the Dynamo install command so you can set the endpoint in one pass:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set-json 'prometheus.prometheusSpec.podMonitorNamespaceSelector={}' \
  --set-json 'prometheus.prometheusSpec.probeNamespaceSelector={}'
```

Then uncomment the `prometheusEndpoint` line in the Dynamo install command. The Dynamo operator automatically creates PodMonitors for its components. See [Metrics](observability/metrics.md) for dashboard setup and available metrics, and [Logging](observability/logging.md) for the Grafana Loki + Alloy logging stack.

### Shared Storage for Model Caching

Set up a `ReadWriteMany` PVC so all pods share downloaded model weights instead of each downloading independently. No Dynamo chart flags are needed — storage is configured in your deployment spec. Setup is cloud-provider-specific:

- [AKS — Azure Files / Managed Lustre](cloud-providers/aks/storage.md)
- [EKS — EFS](cloud-providers/eks/efs.md)
- GKE — Cloud Filestore (see [GKE guide](cloud-providers/gke/gke.md))

For large clusters with frequent model updates, consider [Model Express](model-caching.md#option-2-model-express-p2p-distribution) for P2P model distribution. See [Model Caching](model-caching.md) for the full walkthrough including the download Job and mount configuration.

## Step 4: Pre-Deployment Check

Run the pre-deployment check script to validate your cluster is ready for deployments:

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

This checks kubectl connectivity, default StorageClass configuration, GPU node availability, and GPU Operator status. See [Pre-Deployment Checks](https://github.com/ai-dynamo/dynamo/tree/main/deploy/pre-deployment/README.md) for details.

## Next Steps

Your cluster is ready. Follow the **[Model Deployment Guide](model-deployment-guide.md)** to deploy a model using DGDR.

## Troubleshooting

**"VALIDATION ERROR: Cannot install cluster-wide Dynamo operator"**

```
VALIDATION ERROR: Cannot install cluster-wide Dynamo operator.
Found existing namespace-restricted Dynamo operators in namespaces: ...
```

Cause: Attempting cluster-wide install on a shared cluster with existing namespace-restricted operators.

Solution: Migrate the existing namespace-restricted operators to cluster-wide mode. Namespace-restricted mode is deprecated.

**CRDs already exist**

Cause: Installing CRDs on a cluster where they're already present (common on shared clusters).

Solution: CRDs are installed automatically by the Helm chart. If you encounter conflicts, check existing CRDs with `kubectl get crd | grep dynamo`.

**Pods not starting?**
```bash
kubectl describe pod <pod-name> -n $NAMESPACE
kubectl logs <pod-name> -n $NAMESPACE
```

**Bitnami etcd "unrecognized" image?**

```bash
ERROR: Original containers have been substituted for unrecognized ones.
```

Add to the helm install command:
```bash
--set "etcd.image.repository=bitnamilegacy/etcd" --set "etcd.global.security.allowInsecureImages=true"
```

**Clean uninstall?**

```bash
# Uninstall the platform
helm uninstall dynamo-platform --namespace $NAMESPACE

# List Dynamo CRDs
kubectl get crd | grep "dynamo.*nvidia.com"

# Delete each CRD
kubectl delete crd <crd-name>
```

## Advanced: Build from Source

If you need to contribute to Dynamo or use the latest unreleased features from the main branch:

```bash
# 1. Set registry environment
export DOCKER_SERVER=nvcr.io/nvidia/ai-dynamo/  # or your registry
export DOCKER_USERNAME='$oauthtoken'
export DOCKER_PASSWORD=<YOUR_NGC_CLI_API_KEY>
export IMAGE_TAG=$RELEASE_VERSION

# 2. Build and push operator image
cd deploy/operator
docker build -t $DOCKER_SERVER/kubernetes-operator:$IMAGE_TAG . && docker push $DOCKER_SERVER/kubernetes-operator:$IMAGE_TAG
cd -

# 3. Create namespace and image pull secret (only if using a private registry)
kubectl create namespace $NAMESPACE
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=$DOCKER_SERVER \
  --docker-username=$DOCKER_USERNAME \
  --docker-password=$DOCKER_PASSWORD \
  --namespace=$NAMESPACE

# 4. Install from local chart
cd deploy/helm/charts
helm dep build ./platform/
helm install dynamo-platform ./platform/ \
  --namespace "$NAMESPACE" \
  --set "dynamo-operator.controllerManager.manager.image.repository=$DOCKER_SERVER/kubernetes-operator" \
  --set "dynamo-operator.controllerManager.manager.image.tag=$IMAGE_TAG" \
  --set "dynamo-operator.imagePullSecrets[0].name=docker-imagepullsecret"
```

## Reference

- [Helm Chart Configuration](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/platform/README.md)
- [Create Custom Deployments](./deployment/create-deployment.md)
- [Dynamo Operator Details](./dynamo-operator.md)
- [Model Express Server](https://github.com/ai-dynamo/modelexpress)
