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

<Steps toc={true} tocDepth={2}>

<Step title="Install the GPU Operator">

The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) automates deployment of all NVIDIA software components needed to provision GPUs — drivers, container toolkit, device plugin, and monitoring.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace
  # Note: add \ to --create-namespace above when uncommenting the flag below
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
# Expected: gpu-operator, nvidia-device-plugin-daemonset, etc. all Running
# Note: nvidia-driver-daemonset appears only when the operator manages the driver;
# with provider-managed drivers (--set driver.enabled=false) it is not present.
```

</Step>

<Step title="Install the Dynamo Platform">

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
# Expected: dynamo-operator-* and nats-* pods all Running
```

</Step>

<Step title="Install Optional Components">

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

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) (>= v0.7.0) with [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) for gang scheduling. Both must be installed before deploying multinode workloads.

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

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://github.com/volcano-sh/volcano#quick-start-guide) for configuration options, and the [Multinode Deployment Guide](./deployment/multinode-deployment.md) for orchestrator selection.

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

For large clusters with frequent model updates, consider [ModelExpress](model-caching.md#option-2-modelexpress-p2p-distribution) for P2P model distribution and ModelStreamer for direct streaming from object storage. See [Model Caching](model-caching.md) for the full walkthrough including the download Job, mount configuration, and ModelExpress setup.

</Step>

<Step title="Pre-Deployment Check">

Run the pre-deployment check script to validate your cluster is ready for deployments:

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

This checks kubectl connectivity, default StorageClass configuration, GPU node availability, and GPU Operator status. See [Pre-Deployment Checks](https://github.com/ai-dynamo/dynamo/tree/main/deploy/pre-deployment/README.md) for details.

</Step>

</Steps>


## Alternative: Install with AICR

Steps 1–3 install each component — GPU Operator, Dynamo Platform, Grove, RDMA — with its own Helm command. [NVIDIA AI Cluster Runtime (AICR)](https://github.com/NVIDIA/aicr) is an alternative that generates one version-locked **recipe** for the whole stack, then renders it into deployment-ready bundles for Helm, Argo CD, Flux, or Helmfile. For a supported environment, the `aicr` CLI drives the entire installation flow from a single recipe: GPU Operator, NVIDIA DRA driver, Network Operator (RDMA), Grove, KAI Scheduler, cert-manager, kube-prometheus-stack, and the Dynamo Platform.

> [!NOTE]
> AICR is a new, separately maintained project. It validates **specific combinations** of cloud, GPU, and OS, so it fits a cluster that matches a validated Dynamo recipe rather than an arbitrary environment. You still bring your own Kubernetes cluster — AICR generates and validates the runtime configuration, it does not provision clusters. See the [AICR documentation](https://docs.nvidia.com/aicr) for the authoritative support matrix.

AICR validates recipes across these dimensions; not every combination has a Dynamo recipe, so check the AICR docs for the validated set:

| Dimension | Supported values |
|-----------|------------------|
| Cloud / service | AKS, BCM, EKS, GKE, Kind, LKE, OKE |
| GPU | A100, B200, GB200, H100, H200, L40, RTX PRO 6000 |
| Operating system | Amazon Linux, COS, RHEL, Talos, Ubuntu |

Dynamo recipes use Dynamic Resource Allocation (DRA) for GPUs, which requires **Kubernetes 1.34+**.

### Install Dynamo with the AICR CLI

1. Install the CLI with Homebrew:

```bash
brew tap NVIDIA/aicr
brew install aicr
```

Or use the install script:

```bash
curl -sfL https://raw.githubusercontent.com/NVIDIA/aicr/main/install | bash -s --
```

2. Generate a recipe for your environment. Set `--platform dynamo` and match `--service`, `--accelerator`, and `--os` to your cluster:

```bash
aicr recipe --service eks --accelerator h100 --os ubuntu \
  --intent inference --platform dynamo -o recipe.yaml
```

3. Render the recipe into deployment bundles. Choose the deployer that matches your workflow (`helm`, `argocd`, `flux`, or `helmfile`):

```bash
aicr bundle --recipe recipe.yaml --deployer helm --output ./bundles
```

4. Deploy the generated bundles to your cluster, then validate the running cluster against the recipe:

```bash
aicr validate --recipe recipe.yaml
```

For manual installation, the full command reference, and the supported environment matrix, see the [AICR documentation](https://docs.nvidia.com/aicr). Once AICR has installed the Dynamo Platform, continue with the verification steps above and Next Steps below.

## Next Steps

Your cluster is ready. Follow the **[Deployment Overview](model-deployment-guide.md)** to choose between applying a tuned DGD recipe, creating a DGD directly, or using DGDR to generate one.

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
## Reference

- [Helm Chart Configuration](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/platform/README.md)
- [Create Custom Deployments](./deployment/create-deployment.md)
- [Dynamo Operator Details](./dynamo-operator.md)
- [ModelExpress Server](https://github.com/ai-dynamo/modelexpress)
