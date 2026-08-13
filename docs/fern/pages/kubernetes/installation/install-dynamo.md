---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Installation Guide
subtitle: Install accelerator support and the Dynamo Platform on Kubernetes.
---

This guide walks you through installing everything needed to deploy models with Dynamo on Kubernetes. Follow the steps in order — each builds on the previous one.

## Prerequisites

Before you begin, make sure you have Helm v3 or later and choose the accelerator type in your
cluster:

<Tabs>
<Tab title="NVIDIA GPU">

- A Kubernetes 1.30 or later cluster with NVIDIA GPU nodes. See the cloud provider guides if you need to create one:
  - [Amazon EKS](managed-kubernetes/eks/eks-setup.mdx) | [Azure AKS](managed-kubernetes/azure/aks-setup.mdx) | [Google GKE](managed-kubernetes/gcp/gke-setup.mdx)
  - For local development: [Minikube Setup](../../cli/installation/minikube-setup.mdx)
- `kubectl` 1.30 or later.

> [!IMPORTANT]
> The GPU Operator in the first installation step can install NVIDIA drivers. Do not also enable
> provider-managed drivers when you create the GPU node pool. If the nodes already have drivers,
> disable GPU Operator driver management as shown in that step.

</Tab>
<Tab title="Intel GPU">

- A Kubernetes 1.34 or later cluster with Intel GPU nodes and the `resource.k8s.io/v1` Dynamic Resource Allocation (DRA) API.
- `kubectl` matching the cluster's Kubernetes minor version.
- The [Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes) installed with a `DeviceClass` named `gpu.intel.com`.

> [!IMPORTANT]
> The Dynamo Platform chart does not install the Intel GPU node driver or Intel resource driver.
> Install them before Dynamo workloads request Intel GPUs.

</Tab>
</Tabs>

Verify the client tools:

```bash
kubectl version --client
helm version
```

## Overview

Every Dynamo deployment requires accelerator support and the **Dynamo Platform**. NVIDIA GPU clusters can use the NVIDIA GPU Operator. Intel GPU clusters use the Intel resource driver with DRA. The Dynamo Platform installation is the same after the cluster exposes its accelerator resources. Everything else is optional.

| Optional Component | When you need it | Required for |
|-----------|-----------------|--------------|
| Grove + KAI Scheduler | Multinode or disaggregated inference | Multinode deployments (operator errors without Grove or LWS) |
| Network Operator / RDMA | Disaggregated inference in production | Acceptable KV cache transfer performance (TCP fallback has ~200-500x degradation) |
| kube-prometheus-stack | Autoscaling, metrics dashboards, or the Planner | Planner `sla` mode, KEDA/HPA autoscaling |
| Shared storage (model cache) | Large models (>70B) or many replicas | Avoiding per-pod downloads and HuggingFace rate limits |

**Grove + KAI Scheduler** — Grove is the default multinode orchestrator. The operator returns a hard error on multinode deployments if neither Grove nor [LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws#installation) is available. KAI Scheduler is optional but recommended alongside Grove for GPU-aware scheduling. See [Grove](../../developer-guide/knowledge-base/kubernetes/multinode/grove.md) for details.

**Network Operator / RDMA** — Without RDMA, disaggregated inference falls back to TCP automatically, but with severe performance degradation (~98s TTFT vs ~200-500ms with RDMA). Required for any production disaggregated deployment. Setup is cloud-provider-specific — see the [Disaggregated Communication Guide](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/disagg-communication.md) and your cloud provider guide.

**kube-prometheus-stack** — Required for the Planner's `sla` optimization mode (it reads live TTFT/ITL metrics from Prometheus). Also required for KEDA/HPA-based autoscaling. The Planner's `throughput` mode can function without it using internal queue depth signals, but metrics-driven features will not work. See [Metrics](../operations/observability.mdx) for details.

**Shared storage** — Prevents each pod from downloading model weights independently. Without it, large models (>70B) take hours to download per pod, and many replicas will hit HuggingFace rate limits. Not enforced by the operator — this is an operational concern. See [Model Caching](../model-deployment/model-loading/model-caching.mdx) for the full walkthrough.

<Steps toc={true} tocDepth={2}>

<Step title="Install accelerator support">

<Tabs>
<Tab title="NVIDIA GPU">

The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) automates deployment of the drivers, container toolkit, device plugin, and monitoring components used by NVIDIA GPU nodes.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
  # Add a trailing \ above before uncommenting the following setting.
  # --set driver.enabled=false
```

Set `driver.enabled=false` when the nodes already have provider-managed NVIDIA drivers. See the
[AKS](managed-kubernetes/azure/aks-setup.mdx), [EKS](managed-kubernetes/eks/eks-setup.mdx), or
[GKE](managed-kubernetes/gcp/gke-setup.mdx) guide for provider-specific requirements.

Verify the installation:

```bash
kubectl get pods --namespace gpu-operator
```

</Tab>
<Tab title="Intel GPU">

Install the [Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes) by following the project's installation instructions. The driver must publish a `DeviceClass` named `gpu.intel.com` and `ResourceSlice` objects for the Intel GPU nodes.

Verify the DRA API and Intel devices:

```bash
kubectl api-resources --api-group=resource.k8s.io | grep -E 'deviceclasses|resourceclaims|resourceslices'
kubectl get deviceclass gpu.intel.com
kubectl get resourceslices
```

The Dynamo Platform chart does not install these Intel components. After they are present, install
Dynamo normally in the next step. The operator automatically detects `resource.k8s.io/v1`; no
Dynamo Helm value is required to enable DRA.

For a complete deployment, see [Deploy on Intel GPUs with DRA](../model-deployment/deploy-on-intel-gpus.mdx).

</Tab>
</Tabs>

</Step>

<Step title="Install the Dynamo Platform">

Set your environment variables:

```bash
export NAMESPACE=dynamo-system
export RELEASE_VERSION=1.2.1  # match a version from https://github.com/ai-dynamo/dynamo/releases
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
> **Namespace-restricted mode** (`namespaceRestriction.enabled=true`) is only for development and
> testing. It is not supported for production. Set `dynamo-operator.upgradeCRD=false`; see
> [Dynamo Operator](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/dynamo-operator.md#namespace-restricted-mode).

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

Multinode deployments require either Grove + KAI Scheduler or an alternative orchestrator setup (LeaderWorkerSet + Volcano) to enable gang scheduling for workloads that span multiple nodes. See the [Multinode Deployment Guide](../model-deployment/multinode-deployments.md) for details on orchestrator selection and configuration.

DisaggregatedSet is an opt-in multinode pathway below Grove. To request it, install an LWS release that serves `disaggregatedset.x-k8s.io/v1` and set `nvidia.com/enable-disaggregatedset: "true"` on the DGD. When Grove is available and enabled, also set `nvidia.com/enable-grove: "false"`. Installing the DS API alone does not change existing deployments.

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
> | 1.3.x           | >= v0.13.4    | >= v0.1.0-alpha.8, < v0.1.0-alpha.9 |
> | 1.4.x           | >= v0.13.4    | >= v0.1.0-alpha.10 |
>
> Upgrade Grove in lockstep with Dynamo while the Grove APIs remain unstable. Dynamo 1.3.x expects
> Grove's earlier `ClusterTopology` API and is incompatible with the newer
> `ClusterTopologyBinding` API. Dynamo 1.4.x expects `ClusterTopologyBinding`.

#### LWS + Volcano

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) (>= v0.7.0) with [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) for gang scheduling. Both must be installed before deploying multinode workloads.

1. Install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. Install LWS with Volcano gang scheduling enabled. Use LWS `v0.10.0` or newer when enabling DS; the standard DCD pathway supports LWS `v0.7.0` or newer:

```bash
export LWS_VERSION=0.10.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://github.com/volcano-sh/volcano#quick-start-guide) for configuration options, and the [Multinode Deployment Guide](../model-deployment/multinode-deployments.md) for orchestrator selection.

### Network Operator / RDMA

RDMA setup is cloud-provider-specific. See the [Disaggregated Communication Guide](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/disagg-communication.md) for transport options, UCX configuration, and performance expectations, and your cloud provider guide for setup instructions:

- [AKS — InfiniBand + Network Operator](rdma-setup/infiniband-on-azure.mdx)
- [EKS — EFA device plugin](managed-kubernetes/eks/eks-setup.mdx) (also see the [EFA configuration guide](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/disagg-communication.md#aws-efa-configuration))
- [GKE — GPUDirect-TCPXO](managed-kubernetes/gcp/gke-setup.mdx)

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

Then uncomment the `prometheusEndpoint` line in the Dynamo install command. The Dynamo operator automatically creates PodMonitors for its components. See [Metrics](../operations/observability.mdx) for dashboard setup and available metrics, and [Logging](observability.md) for the Grafana Loki + Alloy logging stack.

### Shared Storage for Model Caching

Set up a `ReadWriteMany` PVC so all pods share downloaded model weights instead of each downloading independently. No Dynamo chart flags are needed — storage is configured in your deployment spec. Setup is cloud-provider-specific:

- [AKS — Azure Files / Managed Lustre](model-storage/aks-storage.md)
- [EKS — EFS](model-storage/efs.mdx)
- GKE — Cloud Filestore (see [GKE guide](managed-kubernetes/gcp/gke-setup.mdx))

For large clusters with frequent model updates, consider [ModelExpress](../model-deployment/model-loading/model-caching.mdx#option-2-modelexpress-p2p-distribution) for P2P model distribution and ModelStreamer for direct streaming from object storage. See [Model Caching](../model-deployment/model-loading/model-caching.mdx) for the full walkthrough including the download Job, mount configuration, and ModelExpress setup.

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

Your cluster is ready. Follow the **[Model Deployment guide](../model-deployment/introduction.mdx)** to choose between applying a tuned DGD recipe, creating a DGD directly, or using DGDR to generate one.

## Troubleshooting

**"VALIDATION ERROR: Cannot install cluster-wide Dynamo operator"**

```
VALIDATION ERROR: Cannot install cluster-wide Dynamo operator.
Found existing namespace-restricted Dynamo operators in namespaces: ...
```

Cause: Attempting cluster-wide install on a shared cluster with existing namespace-restricted operators.

Solution: Remove the development/test namespace-restricted operators, then install one cluster-wide
operator for production use.

**CRDs already exist**

Cause: Installing CRDs on a cluster where they're already present (common on shared clusters).

Solution: The cluster-wide operator's `crd-apply` init container manages CRDs automatically. If you
encounter conflicts, check existing CRDs with `kubectl get crd | grep dynamo`.

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
- [Deploy with DGD](../model-deployment/deploy-with-dgd.md)
- [Dynamo Operator Details](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/dynamo-operator.md)
- [ModelExpress Server](https://github.com/ai-dynamo/modelexpress)
