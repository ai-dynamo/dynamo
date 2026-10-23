---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 安装指南
subtitle: 在 Kubernetes 上安装加速器支持和 Dynamo Platform。
---

本指南介绍在 Kubernetes 上使用 NVIDIA Dynamo 部署模型所需的安装步骤。请按顺序执行，每一步都依赖前一步的配置。

## 前置条件

开始前，请准备 Helm v3 或更新版本，并根据集群中的加速器类型满足以下要求：

<Tabs>
<Tab title="NVIDIA GPU">

- 包含 NVIDIA GPU 节点的 Kubernetes 1.30 或更新版本集群。创建集群可参考以下云平台指南：
  - [Amazon EKS](../../../../../pages/kubernetes/installation/managed-kubernetes/eks/eks-setup.mdx) | [Azure AKS](../../../../../pages/kubernetes/installation/managed-kubernetes/azure/aks-setup.mdx) | [Google GKE](../../../../../pages/kubernetes/installation/managed-kubernetes/gcp/gke-setup.mdx)
  - 本地开发请参考 [Minikube 配置](../../../../../pages/cli/installation/minikube-setup.mdx)
- `kubectl` 1.30 或更新版本。

> [!IMPORTANT]
> 第一步中的 GPU Operator 可以安装 NVIDIA 驱动。创建 GPU 节点池时，不要同时启用云平台托管驱动。如果节点上已有驱动，请按该步骤的说明禁用 GPU Operator 的驱动管理。

</Tab>
<Tab title="Intel GPU">

- 包含 Intel GPU 节点的 Kubernetes 1.34 或更新版本集群，并提供 `resource.k8s.io/v1` 动态资源分配（DRA）API。
- 与集群 Kubernetes minor 版本一致的 `kubectl`。
- 已安装 [Intel Kubernetes 资源驱动](https://github.com/intel/intel-resource-drivers-for-kubernetes)，并存在名为 `gpu.intel.com` 的 `DeviceClass`。

> [!IMPORTANT]
> Dynamo Platform chart 不会安装 Intel GPU 节点驱动或 Intel 资源驱动。Dynamo 工作负载请求 Intel GPU 前，请先安装这些组件。

</Tab>
</Tabs>

验证客户端工具：

```bash
kubectl version --client
helm version
```

## 概述

每个 Dynamo 部署都需要加速器支持和 **Dynamo Platform**。NVIDIA GPU 集群可以使用 NVIDIA GPU Operator；Intel GPU 集群使用 Intel 资源驱动和 DRA。集群提供加速器资源后，Dynamo Platform 的安装流程相同。其余组件均为可选。

| 可选组件 | 使用场景 | 作用 |
|---|---|---|
| Grove + KAI Scheduler | 多节点或分离式推理 | 多节点部署需要 Grove 或 LWS，否则 Operator 会报错 |
| Network Operator / RDMA | 生产环境中的分离式推理 | 保证 KV 缓存传输性能，TCP 回退可能使性能降低约 200–500 倍 |
| kube-prometheus-stack | 自动扩缩容、指标面板或 Planner | Planner 的 `sla` 模式及 KEDA/HPA 自动扩缩容 |
| 共享存储（模型缓存） | 大模型（>70B）或大量副本 | 避免每个 Pod 单独下载及触发 HuggingFace 限流 |

**Grove + KAI Scheduler**：Grove 是默认的多节点编排器。如果 Grove 和 [LeaderWorkerSet（LWS）](https://github.com/kubernetes-sigs/lws#installation) 均不可用，Operator 会拒绝多节点部署。KAI Scheduler 可选，建议与 Grove 配合使用以实现 GPU 感知调度。参阅[多节点编排](multinode-orchestration.md)。

**Network Operator / RDMA**：没有 RDMA 时，分离式推理会自动回退到 TCP，但性能会显著下降，例如首 token 延迟（TTFT）约为 98 秒，而 RDMA 约为 200–500 毫秒。生产环境中的分离式部署需要 RDMA。配置取决于云平台，请参阅 [RDMA 配置](../../../../../pages/kubernetes/installation/rdma-setup/overview.md)及对应云平台指南。

**kube-prometheus-stack**：Planner 的 `sla` 优化模式需要从 Prometheus 读取实时 TTFT 和 token 间延迟（ITL）指标；KEDA/HPA 自动扩缩容也需要该组件。Planner 的 `throughput` 模式可使用内部队列深度信号运行，但依赖指标的功能不可用。参阅[指标](../../../../../pages/kubernetes/operations/observability.mdx)。

**共享存储**：避免每个 Pod 独立下载模型权重。缺少共享存储时，大模型（>70B）在每个 Pod 上可能需要数小时下载，大量副本也可能触发 HuggingFace 限流。Operator 不强制要求共享存储，需要由运维人员规划。参阅[模型存储概述](../../../../../pages/kubernetes/installation/model-storage/overview.md)以选择存储后端。

<Steps toc={true} tocDepth={2}>

<Step title="安装加速器支持">

<Tabs>
<Tab title="NVIDIA GPU">

[NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) 自动部署 NVIDIA GPU 节点使用的驱动、容器工具包、设备插件和监控组件。

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
  # Add a trailing \ above before uncommenting the following setting.
  # --set driver.enabled=false
```

节点已有云平台托管的 NVIDIA 驱动时，设置 `driver.enabled=false`。平台要求请参阅 [AKS](../../../../../pages/kubernetes/installation/managed-kubernetes/azure/aks-setup.mdx)、[EKS](../../../../../pages/kubernetes/installation/managed-kubernetes/eks/eks-setup.mdx) 或 [GKE](../../../../../pages/kubernetes/installation/managed-kubernetes/gcp/gke-setup.mdx) 指南。

验证安装结果：

```bash
kubectl get pods --namespace gpu-operator
```

</Tab>
<Tab title="Intel GPU">

按照 [Intel Kubernetes 资源驱动](https://github.com/intel/intel-resource-drivers-for-kubernetes)的安装说明部署驱动。驱动必须提供名为 `gpu.intel.com` 的 `DeviceClass`，以及描述 Intel GPU 节点的 `ResourceSlice` 对象。

验证 DRA API 和 Intel 设备：

```bash
kubectl api-resources --api-group=resource.k8s.io | grep -E 'deviceclasses|resourceclaims|resourceslices'
kubectl get deviceclass gpu.intel.com
kubectl get resourceslices
```

Dynamo Platform chart 不会安装这些 Intel 组件。组件就绪后，按下一步正常安装 Dynamo。Operator 自动检测 `resource.k8s.io/v1`，无需设置 Dynamo Helm 参数来启用 DRA。

完整部署流程请参阅[通过 DRA 在 Intel GPU 上部署](../../../../../pages/kubernetes/model-deployment/deploy-on-intel-gpus.mdx)。

</Tab>
</Tabs>

</Step>

<Step title="安装 Dynamo Platform">

设置环境变量：

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
  # --set-string "kai-scheduler.scheduler.args.default-staleness-grace-period=-1s" \
  # Option B (enabled=true): Grove/KAI are already installed externally (production):
  # --set "global.grove.enabled=true" \
  # --set "global.kai-scheduler.enabled=true" \
  #
  # kube-prometheus-stack — uncomment if Prometheus is installed (required for Planner sla mode and autoscaling):
  # --set "dynamo-operator.dynamo.metrics.prometheusEndpoint=http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
```

> [!TIP]
> 所有 `helm install` 命令都可以通过自定义 values 文件配置：`helm install ... -f your-values.yaml`。

> [!WARNING]
> **实验性功能，仅用于开发集群。** 尚未完成的 DGDR v1beta2 schema 默认禁用。
> 上面使用的 `1.2.1` chart 不支持此功能开关。在正式支持此 API 的版本发布前，
> 如需测试，请使用基于当前 `main` 分支构建的开发版 chart，并在 Helm 命令中添加
> `--set "dynamo-operator.dgdrV1Beta2.enabled=true"`。
> 该参数添加 v1beta2 `DynamoGraphDeploymentRequest` schema，并安装
> `DynamoGraphDeploymentRun` 和 `DynamoGraphDeploymentCandidate` 自定义资源定义（CRD）。
> 它不会启用 v1beta2 协调逻辑，Operator 不会处理使用这些 schema 创建的资源。
>
> Kubernetes 会将 Run 和 Candidate 对象持久化到 etcd。schema 尚未定稿，后续变更可能导致
> 已有对象无法读取或阻止 CRD 升级。不要在生产或共享集群启用此 API。

> [!TIP]
> **共享或多租户集群**：如果已有集群级 Dynamo Operator 运行，请不要再安装另一个。通过以下命令检查：
> ```bash
> kubectl get clusterrolebinding -o json | \
>   jq -r '.items[] | select(.metadata.name | contains("dynamo-operator-manager")) |
>   "Cluster-wide operator found in namespace: \(.subjects[0].namespace)"'
> ```

> [!WARNING]
> **命名空间受限模式**（`namespaceRestriction.enabled=true`）仅用于开发和测试，不支持生产环境。启用该模式时，请设置 `dynamo-operator.upgradeCRD=false`。

验证 Dynamo Platform 已正常运行：

```bash
# Check CRDs
kubectl get crd | grep dynamo
# Expected: dynamographdeployments, dynamocomponentdeployments, dynamographdeploymentrequests, etc.

# Check operator and platform pods
kubectl get pods -n $NAMESPACE
# Expected: dynamo-operator-* and nats-* pods all Running
```

</Step>

<Step title="安装可选组件">

上面的 Dynamo 安装命令包含各可选组件的注释参数。先安装对应组件，再取消相关参数的注释，运行第二步中的 `helm install`。如果 Dynamo 已安装，可使用 `helm upgrade --reuse-values` 并添加对应参数。

### 多节点部署

多节点部署需要编排器来管理跨节点工作负载。强烈建议使用 gang scheduling，将所需的所有 Pod 一起调度，避免部分资源分配导致工作负载停滞。编排器选择和配置请参阅[多节点编排](multinode-orchestration.md)。

DisaggregatedSet 是优先级低于 Grove 的可选多节点路径。新建 DGD 时，先安装提供 `disaggregatedset.x-k8s.io/v1` API 的 LWS，再设置 `nvidia.com/enable-disaggregatedset: "true"`。如果 Grove 可用且已启用，还需设置 `nvidia.com/enable-grove: "false"`。Dynamo 将所选编排器持久化到 `nvidia.com/workload-provider`。安装 DS API 或修改启用注解不会改变已有部署的编排器。如果所选编排器不可用，Dynamo 会报告 `Ready=False`，不会自动切换。需要切换路径时，请删除并重新创建 DGD。

以下各节介绍如何为各个支持的编排器启用 gang scheduling。

#### Grove + KAI Scheduler

在 Dynamo 安装命令中取消不同参数的注释，可以选择以下两种方式启用 Grove 和 KAI Scheduler：

- **`install=true`**：Dynamo 通过捆绑的子 chart 安装并管理 Grove/KAI，适用于开发和测试。
- **`enabled=true`**：告知 Dynamo，Grove/KAI 已经安装并由外部管理。需要单独管理生命周期或跨命名空间共享组件时使用，推荐用于生产环境。

使用 `enabled=true` 时，先分别安装 Grove 和 KAI Scheduler。参阅 [Grove 安装指南](https://github.com/NVIDIA/grove/blob/main/docs/installation.md)和 [KAI Scheduler 部署指南](https://github.com/NVIDIA/KAI-Scheduler)。

> [!NOTE]
> **兼容性矩阵：**
>
> | dynamo-platform | kai-scheduler | Grove |
> |-----------------|---------------|-------|
> | 1.0.x           | >= v0.13.0    | >= v0.1.0-alpha.6 |
> | 1.1.x           | >= v0.13.4    | >= v0.1.0-alpha.8 |
> | 1.3.x           | >= v0.13.4    | >= v0.1.0-alpha.8, < v0.1.0-alpha.9 |
> | 1.4.x           | >= v0.13.4    | >= v0.1.0-alpha.10 |
> | 1.5.x           | >= v0.17.0    | >= v0.1.0-alpha.13 |
> | 1.6.x           | >= v0.17.0    | >= v0.1.0-alpha.15 |
>
> Grove API 尚未稳定时，请让 Grove 与 Dynamo 同步升级。Dynamo 1.3.x 使用较早的 `ClusterTopology` API，与较新的 `ClusterTopologyBinding` API 不兼容；Dynamo 1.4.x 使用 `ClusterTopologyBinding`。

#### LWS + Volcano

如果不使用 Grove 进行多节点编排，可以结合 [LeaderWorkerSet（LWS）](https://lws.sigs.k8s.io/docs/installation/)（>= v0.7.0）和 [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) 实现 gang scheduling。在部署多节点工作负载前，必须先安装这两个组件。

1. 安装 Volcano：

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. 安装 LWS 并启用 Volcano gang scheduling。启用 DS 时，请使用提供 `disaggregatedset.x-k8s.io/v1` API 的 LWS 版本（当前 Dynamo Operator 依赖 LWS `v0.10.0`）；标准 DCD 路径支持 LWS `v0.7.0` 及以上版本：

```bash
export LWS_VERSION=0.10.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

配置选项请参阅 [LWS 文档](https://lws.sigs.k8s.io/docs/)和 [Volcano 文档](https://github.com/volcano-sh/volcano#quick-start-guide)。各编排器的比较请参阅[多节点编排](multinode-orchestration.md)。

### Network Operator / RDMA

RDMA 配置取决于云平台。要求和平台操作步骤请参阅 [RDMA 配置](../../../../../pages/kubernetes/installation/rdma-setup/overview.md)：

- [AKS：InfiniBand + Network Operator](../../../../../pages/kubernetes/installation/rdma-setup/infiniband-on-azure.mdx)
- [EKS：EFA 设备插件](../../../../../pages/kubernetes/installation/managed-kubernetes/eks/eks-setup.mdx)，另请参阅 [AWS 上的 EFA](../../../../../pages/kubernetes/installation/rdma-setup/efa-on-aws.mdx)
- [GKE：GPUDirect-TCPXO](../../../../../pages/kubernetes/installation/managed-kubernetes/gcp/gke-setup.mdx)

### kube-prometheus-stack

先安装 Prometheus，再运行 Dynamo 安装命令，以便同时配置指标端点：

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set-json 'prometheus.prometheusSpec.podMonitorNamespaceSelector={}' \
  --set-json 'prometheus.prometheusSpec.probeNamespaceSelector={}'
```

然后取消 Dynamo 安装命令中 `prometheusEndpoint` 所在行的注释。Dynamo Operator 会自动为组件创建 PodMonitor。指标和面板配置请参阅[指标](../../../../../pages/kubernetes/operations/observability.mdx)；Grafana Loki + Alloy 日志栈请参阅[日志](../../../../../pages/reference/observability/logging.mdx)。

### 模型缓存共享存储

配置 `ReadWriteMany` PVC，让所有 Pod 共享已下载的模型权重，避免重复下载。无需设置 Dynamo chart 参数，存储通过部署 spec 配置。操作取决于云平台：

- [AKS：Azure Files / Managed Lustre](../../../../../pages/kubernetes/installation/model-storage/aks-storage.md)
- [EKS：EFS](../../../../../pages/kubernetes/installation/model-storage/efs.mdx)
- GKE：Cloud Filestore，参阅 [GKE 指南](../../../../../pages/kubernetes/installation/managed-kubernetes/gcp/gke-setup.mdx)

对于模型更新频繁的大型集群，可考虑使用 [ModelExpress](../../../../../pages/kubernetes/installation/model-storage/overview.md) 实现 P2P 模型分发，或使用 ModelStreamer 直接从对象存储流式加载。下载 Job、挂载配置及 ModelExpress 安装流程请参阅[模型缓存](../../../../../pages/kubernetes/installation/model-storage/overview.md)。

</Step>

<Step title="部署前检查">

运行部署前检查脚本，验证集群是否已具备部署条件：

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

该脚本检查 kubectl 连通性、默认 StorageClass 配置、GPU 节点可用性和 GPU Operator 状态。参阅[部署前检查](https://github.com/ai-dynamo/dynamo/tree/main/deploy/pre-deployment/README.md)。

</Step>

</Steps>


## 替代方案：通过 AICR 安装

前面的一到三步分别通过 Helm 安装 GPU Operator、Dynamo Platform、Grove 和 RDMA。[NVIDIA AI Cluster Runtime（AICR）](https://github.com/NVIDIA/aicr) 提供另一种方式：为整套组件生成锁定版本的 **recipe**，再渲染为适用于 Helm、Argo CD、Flux 或 Helmfile 的部署包。对于受支持的环境，`aicr` CLI 可以通过一个 recipe 驱动 GPU Operator、NVIDIA DRA driver、Network Operator（RDMA）、Grove、KAI Scheduler、cert-manager、kube-prometheus-stack 和 Dynamo Platform 的安装。

> [!NOTE]
> AICR 是独立维护的新项目，只验证特定的云平台、GPU 和操作系统组合。请确认集群符合已验证的 Dynamo recipe。Kubernetes 集群需要自行准备，AICR 生成并验证运行时配置，不负责创建集群。支持矩阵以 [AICR 文档](https://docs.nvidia.com/aicr)为准。

AICR 从以下维度验证 recipe。并非每种组合都有 Dynamo recipe，已验证的组合请参阅 AICR 文档：

| 维度 | 支持的值 |
|---|---|
| 云平台 / 服务 | AKS、BCM、EKS、GKE、Kind、LKE、OKE |
| GPU | A100、B200、GB200、H100、H200、L40、RTX PRO 6000 |
| 操作系统 | Amazon Linux、COS、RHEL、Talos、Ubuntu |

Dynamo recipe 通过动态资源分配（DRA）使用 GPU，因此需要 **Kubernetes 1.34 或更新版本**。

### 使用 AICR CLI 安装 Dynamo

1. 使用 Homebrew 安装 CLI：

```bash
brew tap NVIDIA/aicr
brew install aicr
```

也可以使用安装脚本：

```bash
curl -sfL https://raw.githubusercontent.com/NVIDIA/aicr/main/install | bash -s --
```

2. 为环境生成 recipe。设置 `--platform dynamo`，并让 `--service`、`--accelerator` 和 `--os` 与集群匹配：

```bash
aicr recipe --service eks --accelerator h100 --os ubuntu \
  --intent inference --platform dynamo -o recipe.yaml
```

3. 将 recipe 渲染为部署包。选择符合工作流的部署工具：`helm`、`argocd`、`flux` 或 `helmfile`。

```bash
aicr bundle --recipe recipe.yaml --deployer helm --output ./bundles
```

4. 将生成的部署包应用到集群，再根据 recipe 验证运行中的集群：

```bash
aicr validate --recipe recipe.yaml
```

手动安装方法、完整命令参考和支持环境矩阵请参阅 [AICR 文档](https://docs.nvidia.com/aicr)。AICR 安装 Dynamo Platform 后，请继续执行上面的验证步骤及下方的后续步骤。

## 后续步骤

集群已准备就绪。参阅[模型部署指南](../../../../../pages/kubernetes/model-deployment/introduction.mdx)，选择使用经过调优的 DGD recipe、直接创建 DGD，或通过 DGDR 生成部署配置。

## 故障排查

**"VALIDATION ERROR: Cannot install cluster-wide Dynamo operator"**

```
VALIDATION ERROR: Cannot install cluster-wide Dynamo operator.
Found existing namespace-restricted Dynamo operators in namespaces: ...
```

原因：共享集群中存在命名空间受限的 Operator，但当前尝试安装集群级 Operator。

解决方法：删除开发或测试用途的命名空间受限 Operator，再安装一个用于生产环境的集群级 Operator。

**CRD 已存在**

原因：集群中已存在相关 CRD，共享集群中较为常见。

解决方法：集群级 Operator 的 `crd-apply` 初始化容器会自动管理 CRD。如遇冲突，运行 `kubectl get crd | grep dynamo` 检查已有 CRD。

**Pod 无法启动？**
```bash
kubectl describe pod <pod-name> -n $NAMESPACE
kubectl logs <pod-name> -n $NAMESPACE
```

**Bitnami etcd 提示镜像无法识别？**

```bash
ERROR: Original containers have been substituted for unrecognized ones.
```

在 Helm 安装命令中添加：
```bash
--set "etcd.image.repository=bitnamilegacy/etcd" --set "etcd.global.security.allowInsecureImages=true"
```

**如何完整卸载？**

```bash
# Uninstall the platform
helm uninstall dynamo-platform --namespace $NAMESPACE

# List Dynamo CRDs
kubectl get crd | grep "dynamo.*nvidia.com"

# Delete each CRD
kubectl delete crd <crd-name>
```
## 参考资料

- [Helm Chart 配置](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/platform/README.md)
- [通过 DGD 部署](../../../../../pages/kubernetes/model-deployment/deploy-with-dgd.md)
- [Kubernetes API 参考](../../../../../pages/reference/kubernetes-api/full-api-reference.mdx)
- [ModelExpress 服务器](https://github.com/ai-dynamo/modelexpress)
