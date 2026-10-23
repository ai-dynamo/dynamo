---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 使用 DGDR 自动部署
subtitle: 描述模型、工作负载和 SLA 目标，让 Dynamo 分析硬件并生成 DynamoGraphDeployment。
---

**DynamoGraphDeploymentRequest（DGDR）** 是 NVIDIA Dynamo 按意图部署模型的方式。无需手动编写包含并行度、副本数和资源限制的 [DynamoGraphDeployment（DGD）](../../../../../pages/kubernetes/model-deployment/deploy-with-dgd.md)，只需描述要运行的模型、后端、工作负载和可选延迟目标。Dynamo 的 profiler 会分析集群 GPU，选择配置并生成提供推理服务的 DGD。

本指南从最小 DGDR 开始，逐步添加工作负载目标、搜索策略、硬件规模、模型缓存、运行时自动扩缩容和部署前审核。组件关系及 DGDR 适用场景请参阅[自动部署概述](../../../../../pages/kubernetes/auto-deployment/overview.mdx)；完整字段表和生命周期请参阅 [DGDR 参考](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx)；可直接使用的清单请参阅 [DGDR 模板](../../../../../pages/recipes/kubernetes-templates/dgdr.mdx)。

> [!NOTE]
> 使用正式发布版本安装时，如果省略 `spec.image`，DGDR webhook 会选择
> `nvcr.io/nvidia/ai-dynamo/dynamo-planner:<operator-version>`。
> 对于版本未知的本地 Operator 构建，请显式设置 `spec.image`。
> 使用与 Operator 同一发布版本的 `dynamo-planner` 镜像。
> 对于早于 1.1.0 的 Dynamo 版本，使用对应版本的 `dynamo-frontend` 镜像。
> 生成的 Planner 和后端镜像会保留相同的镜像仓库和标签。

## 前置条件

编写 DGDR 前，请准备：

- 已安装 **Dynamo Platform** 的 Kubernetes 集群，包括 Operator 和 profiler。请参阅[安装指南](../installation/install-dynamo.md)。
- 该集群的 `kubectl` 访问权限及目标命名空间。
- Operator 可通过 DCGM 或节点标签发现的 GPU 节点。Operator 自动检测 SKU、显存和 GPU 数量；这些值均可覆盖，见[步骤四](#设置硬件并处理大型或多节点模型)。
- 命名空间中名为 `hf-token-secret` 的 **HuggingFace token Secret**，用于需要授权或受速率限制的模型。性能分析 Job 和部署的 Pod 都会使用它。
- 首次在 Kubernetes 上使用 Dynamo 时，先按 [Kubernetes 快速入门](../../../../../pages/kubernetes/getting-started/quickstart.mdx)完整运行一个模型。

如果需要 SLA 驱动的自动扩缩容，还需在创建 DGDR 前安装 [Prometheus](../installation/install-dynamo.md#kube-prometheus-stack)，见[步骤六](#启用-planner-进行运行时自动扩缩容)。

## DGDR 结构

DGDR 是一份简短的意图描述。唯一必填字段是 `model`，其余字段都有默认值或可自动检测。profiler 会补全省略的部分。

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B          # HuggingFace model ID (required)
  backend: auto                   # auto | vllm | sglang | trtllm
  searchStrategy: rapid           # rapid (~30s, simulated) | thorough (2–4h, real GPU)
  autoApply: true                 # deploy the generated DGD automatically
  # workload: { ... }             # expected traffic shape (Step 2)
  # sla: { ... }                  # latency targets (Step 2)
  # hardware: { ... }             # override auto-detected GPUs (Step 4)
  # modelCache: { ... }           # mount cached weights (Step 5)
  # features: { planner: { ... } }# runtime autoscaling (Step 6)
  # overrides: { ... }            # customize the generated DGD (Step 7)
```

以下步骤逐步填充这些字段，以满足更复杂的部署需求。完整字段说明和默认值请参阅 [DGDR 字段参考](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx#spec-reference)。

<Steps toc={true} tocDepth={2}>

<Step title="提交最小 DGDR">

从只包含模型的最小请求开始。`searchStrategy` 和 `autoApply` 使用默认值时，profiler 执行快速模拟，约需 30 秒，分析期间不占用 GPU，随后由 Operator 部署结果。使用正式发布版本安装时，webhook 会选择对应版本的 Planner 镜像。

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-small
spec:
  model: Qwen/Qwen3-0.6B
```

应用清单并观察请求的阶段变化：

```bash
kubectl apply -f qwen-small.yaml -n <namespace>
kubectl get dgdr qwen-small -n <namespace> -w
```

DGDR 依次经过 `Pending` → `Profiling` → `Ready` → `Deploying` → `Deployed`。到达 `Deployed` 时，生成的 DGD 已在运行。完整阶段列表和日志命令请参阅[监控性能分析和部署](#监控性能分析和部署)。

> [!TIP]
> 生成的 DGD **不会**设置指向 DGDR 的 owner reference。删除 DGDR 后，DGD 仍会提供服务。
> 如需完整清理，请同时删除两者，见[清理资源](#清理资源)。

</Step>

<Step title="描述工作负载和 SLA 目标">

profiler 根据预期流量和延迟要求确定部署规模。提供 `workload`，并按需添加 `sla` 目标，让 profiler 针对实际场景优化，而不是使用默认值。

```yaml
spec:
  model: meta-llama/Llama-3.1-8B-Instruct
  backend: vllm
  workload:
    isl: 4000          # average input sequence length (tokens)
    osl: 1000          # average output sequence length (tokens)
    requestRate: 10    # target requests per second
  sla:
    ttft: 500          # target Time To First Token (ms)
    itl: 50            # target Inter-Token Latency (ms)
```

| 字段 | 含义 | 默认值 |
|---|---|---|
| `workload.isl` | 预期平均输入序列长度 | `4000` |
| `workload.osl` | 预期平均输出序列长度 | `1000` |
| `workload.requestRate` | 目标每秒请求数 | — |
| `workload.concurrency` | 目标并发请求数，可替代 `requestRate` | — |
| `sla.ttft` | 首 token 时间目标，单位 ms | `2000` |
| `sla.itl` | token 间延迟目标，单位 ms | `30` |
| `sla.e2eLatency` | 端到端延迟目标，单位 ms。**不能**与 `ttft`/`itl` 同时使用。 | — |

> [!NOTE]
> SLA 目标影响 profiler 的配置选择，也是 [Planner](#启用-planner-进行运行时自动扩缩容)
> 在运行时努力达到的目标。请设置可实现的目标；无法达到的值会使 profiler 选择成本最高的布局。

</Step>

<Step title="选择搜索策略">

`searchStrategy` 控制 profiler 如何探索配置，需要在分析时间和配置的优化程度之间取舍。

**Rapid（默认）** 使用基于 AIC 的 DynoSim 风格性能建模搜索配置，无需运行真实推理。约 30 秒完成，分析期间不占用 GPU。

```yaml
spec:
  searchStrategy: rapid
```

入门、快速迭代或在 CI/CD 中运行时可使用 rapid，但 GPU SKU 需要位于 [profiler 支持矩阵](../../../../../pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md#support-matrix)中。如果 AIC 不支持模型、硬件与后端的组合，profiler 会回退到简单的显存适配配置，其性能不一定最优。

**Thorough** 枚举候选并行配置，在真实 GPU 上逐个部署，并使用 AIPerf 进行基准测试。耗时 2–4 小时，产生实测数据而非模拟数据。

```yaml
spec:
  searchStrategy: thorough
  backend: vllm      # must specify a concrete backend
```

生产调优、AIC 不支持的硬件（如 PCIe GPU），或需要实测性能时，使用 thorough。它有以下限制：

- **仅支持分离式模式**：thorough 不运行聚合式配置。
- **不接受 `backend: auto`**：必须指定 `vllm`、`sglang` 或 `trtllm`。
- **需要 GPU 资源**：profiler 在分析期间部署真实推理引擎。

性能分析算法、门控检查和模式选择请参阅 [profiler 指南](../../../../../pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md)。

</Step>

<Step title="设置硬件并处理大型或多节点模型">

Operator 自动检测 GPU SKU、显存和数量，数量上限为 32。如果自动检测结果有误，或希望 profiler 考虑更多 GPU，请在 `hardware` 中覆盖相应字段。

```yaml
spec:
  hardware:
    gpuSku: h200_sxm   # lowercase underscore format
    totalGpus: 64      # raise above the 32-GPU auto-detect cap
    numGpusPerNode: 8
```

GPU SKU 使用**小写字母加下划线**格式，例如 `h100_sxm`，而非 `H100-SXM5-80GB`。完整取值列表和 PCIe 支持限制请参阅 [DGDR SKU 格式参考](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx#sku-format)。

**跨节点的大型模型和 MoE 模型。** 当模型需要的 GPU 数超过单节点容量时，部署需要 Grove 或 LWS 等多节点编排器，并强烈建议使用 gang scheduling。安装前置条件请参阅[多节点编排](../installation/multinode-orchestration.md)。


对于 **Mixture-of-Experts（MoE）** 模型（DeepSeek-R1、Qwen3-MoE），使用 **SGLang** 可获得完整支持；vLLM 和 TensorRT-LLM 的 MoE 支持仍在开发中，目前仅部分可用。profiler 最多在 **4 个节点**上搜索 MoE 配置；超过此范围时，会选择范围内的最佳配置，可能需要手动调整副本数。请参阅 [profiler 支持矩阵](../../../../../pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md#support-matrix)。

> [!NOTE]
> DGDR 不暴露 `multinode.nodeCount`。`hardware.totalGpus` 设置性能分析和部署的总 GPU 预算，
> `hardware.numGpusPerNode` 描述每个节点的容量。选择各 worker 的并行配置后，profiler
> 将单实例 GPU 数除以 `numGpusPerNode` 并向上取整，作为生成 DGD 的 `multinode.nodeCount`。
> worker 可在单节点内容纳时，不生成多节点配置。

</Step>

<Step title="缓存模型权重">

默认情况下，性能分析 Job 和每个生成的 worker Pod 都从 HuggingFace 下载模型。大型模型（>70B）下载缓慢，多个副本还可能触发 HuggingFace 速率限制。将 `modelCache` 指向已填充权重的 `ReadWriteMany` PVC，即可从共享存储加载。

```yaml
spec:
  model: meta-llama/Llama-3.1-70B-Instruct
  modelCache:
    pvcName: model-cache
    pvcMountPath: /home/dynamo/.cache/huggingface
    pvcModelPath: hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/<commit-hash>
```

Operator 以只读方式将 PVC 挂载到性能分析 Job，并传递给生成的 DGD，让性能分析和推理服务都使用缓存权重。

`pvcModelPath` 必须是 PVC 内的 HuggingFace 快照路径：`hub/models--<org>--<model>/snapshots/<commit-hash>`。将模型 ID 中的 `/` 替换为 `--`，并将 `<commit-hash>` 替换为实际快照版本。

**配置步骤：** 创建 `ReadWriteMany` PVC，参阅[安装指南：共享存储](../installation/install-dynamo.md#模型缓存共享存储)；运行一次下载 Job 填充权重，然后在此引用 PVC。存储后端选择请参阅[模型存储概述](../../../../../pages/kubernetes/installation/model-storage/overview.md)。

对于需要授权的模型，创建 profiler 和 Pod 会自动读取的 token Secret：

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-token> \
  -n <namespace>
```

</Step>

<Step title="启用 Planner 进行运行时自动扩缩容">

**Planner** 为分离式部署提供运行时自动扩缩容：流量变化时，调整 prefill 和 decode 副本数，以满足 SLA 目标。将 `features.planner` 设置为 PlannerConfig 即可启用。DGDR 将该对象传递给 Planner 服务，并在最终 DGD 中生成 Planner 支持配置。

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  features:
    planner:
      mode: disagg
      backend: vllm
  sla:
    ttft: 500
    itl: 50
```

如果只想评估 Planner 建议而不执行扩缩容，在同一对象中添加 `advisory: true`。

```yaml
spec:
  features:
    planner:
      mode: disagg
      backend: vllm
      advisory: true
```

Planner 的 `sla` 优化目标从 Prometheus 读取实时 TTFT/ITL。因此，需要 SLA 驱动扩缩容时，请先安装 [Prometheus](../installation/install-dynamo.md#kube-prometheus-stack)，再创建 DGDR。`throughput` 和 `latency` 模式使用内部队列深度信号，无需 Prometheus。扩缩容模式和完整 PlannerConfig 字段请参阅 [Planner 指南](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md)。

> [!NOTE]
> `features.planner` 本身就是 PlannerConfig 对象，不使用 `enabled` 包装字段。
> DGDR API 直接传递该对象，不进行字段级校验。profiler 会复制对象，添加所选配置的 GPU 数量，
> 并传递 DGDR 显式设置的延迟目标，除非 `features.planner.ttft_ms` 或
> `features.planner.itl_ms` 覆盖它们。随后将协调后的对象写入 ConfigMap 中的
> `planner_config.json`，把文件挂载到生成的 Planner 服务，并通过 `--config` 启动服务。
> 非 advisory 模式下，还会为 `mode` 选中的 worker 服务启用扩缩容适配器。

</Step>

<Step title="审核生成的 DGD">

生产环境中，请在部署前检查生成的 DGD。设置 `autoApply: false`，使 DGDR 停留在 `Ready` 阶段并保存配置，而不直接部署。

```yaml
spec:
  autoApply: false
```

性能分析完成后，自行提取、审核并应用 DGD：

```bash
kubectl get dgdr my-model -n <namespace> \
  -o jsonpath='{.status.profilingResults.selectedConfig}' > my-dgd.yaml
# Review and edit my-dgd.yaml, then:
kubectl apply -f my-dgd.yaml -n <namespace>
```

禁用 `autoApply` 时，可以在 `Profiling` 或 `Ready` 阶段设置或修改 `runtimeVersionOverride`。如需让 Operator 部署审核后的快照，请启用 `autoApply`：

```bash
kubectl patch dgdr my-model -n <namespace> --type=merge \
  -p '{"spec":{"runtimeVersionOverride":"1.6.0"}}'
kubectl patch dgdr my-model -n <namespace> --type=merge \
  -p '{"spec":{"autoApply":true}}'
```

也可以在启用 `autoApply` 的同一次更新中设置或修改该覆盖值。Operator 会将当前值应用到新 DGD，不修改已保存的快照。

</Step>

</Steps>

## 监控性能分析和部署

DGDR 依次经过以下阶段。性能分析失败属于终止状态，不会重试（`backoffLimit: 0`）。

| 阶段 | 当前操作 |
|---|---|
| `Pending` | spec 已通过校验；Operator 正在发现 GPU 硬件并准备性能分析 Job |
| `Profiling` | 性能分析 Job 运行中，子阶段包括 `Initializing`、`SweepingPrefill`、`SweepingDecode`、`SelectingConfig`、`BuildingCurves`、`GeneratingDGD`、`Done` |
| `Ready` | 性能分析完成，配置保存在 `.status.profilingResults.selectedConfig`。等待手动应用或启用 `autoApply`。 |
| `Deploying` | 正在创建 DGD，仅在 `autoApply: true` 时执行 |
| `Deployed` | DGD 正在运行且健康 |
| `Failed` | 不可恢复的错误，请检查事件和 condition |

观察进度并读取性能分析日志：

```bash
NAMESPACE=your-namespace

# Watch phase transitions
kubectl get dgdr my-model -n "$NAMESPACE" -w

# Detailed status, conditions, and events
kubectl describe dgdr my-model -n "$NAMESPACE"

# Current profiling sub-phase
kubectl get dgdr my-model -n "$NAMESPACE" -o jsonpath='{.status.profilingPhase}'

# Profiling job logs
PROFILING_JOB=$(kubectl get dgdr my-model -n "$NAMESPACE" -o jsonpath='{.status.profilingJobName}')
kubectl logs -f "job/${PROFILING_JOB}" -c profiler -n "$NAMESPACE"
```

完整生命周期、condition 和监控命令请参阅 [DGDR 生命周期参考](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx#lifecycle)。

## 故障排查

| 现象 | 原因与解决方法 |
|---|---|
| **性能分析或推理服务期间 OOM** | 模型无法放入所选 TP 配置的 GPU 显存。增加 `hardware.totalGpus`；长上下文、KV 开销等情况需要超过最低配置的资源。 |
| **profiler 忽略额外 GPU** | 自动检测上限为 32。显式设置 `hardware.totalGpus`。 |
| **性能分析 Job 无法调度** | GPU 节点存在污点。通过 `overrides.profilingJob` 添加容忍。 |
| **spec 修改被拒绝** | DGDR 进入 `Profiling` 后 spec 不可变，但 `autoApply: false` 的请求可在 `Profiling` 或 `Ready` 阶段设置或修改 `runtimeVersionOverride`，并可在 `Ready` 阶段启用 `autoApply`。其他修改需要删除并重新创建 DGDR。 |
| **多节点部署报错** | 未安装 Grove 或 LWS。请参阅[多节点编排](../installation/multinode-orchestration.md)。 |

### 性能分析 Job 无法调度

GPU 节点通常带有污点。通过 `overrides.profilingJob` 为生成的性能分析 Job 添加所需容忍：

```yaml
spec:
  overrides:
    profilingJob:
      template:
        spec:
          containers: []
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
```

空的 `containers` 列表是必需的占位符。Operator 将覆盖配置合并到其为性能分析 Job 生成的容器配置中。

## 清理资源

删除 DGDR **不会**删除其创建的 DGD，DGD 会继续提供服务。如需同时删除两者：

```bash
kubectl delete dgdr my-model -n <namespace>
kubectl delete dgd my-model-dgd -n <namespace>
```

> [!NOTE]
> 生成的 DGD 默认名为 `<dgdr-name>-dgd`。显式设置 `spec.overrides.dgd.metadata.name`
> 会替换默认值。可运行 `kubectl get dgd -n <namespace> -l dgdr.nvidia.com/name=my-model`
> 查找 DGD。所选名称也记录在 `.status.dgdName` 中。

## 可选：自定义生成的 DGD

生成的 DGD 需要设置 DGDR 未直接暴露的字段时，使用 `spec.overrides.dgd`。提供部分 `nvidia.com/v1beta1` DGD。profiler 选择配置后，DGDR 按 `name` 合并匹配的组件、容器和环境变量。

例如，为生成的 `Frontend` 组件启用 KV 感知路由：

```yaml
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        components:
        - name: Frontend
          podTemplate:
            spec:
              containers:
              - name: main
                env:
                - name: DYN_ROUTER_MODE
                  value: kv
```

在性能分析开始前添加覆盖配置，使用所选后端和拓扑生成的精确组件名，区分大小写。可按以下步骤查找某个 Dynamo 版本的组件名称：

1. 创建临时 DGDR，设置 `autoApply: false`，不添加组件覆盖配置。
2. 等待 DGDR 到达 `Ready` 阶段，再列出生成的名称：

```bash
kubectl get dgdr <name> -n <namespace> \
  -o jsonpath='{range .status.profilingResults.selectedConfig.spec.components[*]}{.name}{"\n"}{end}'
```

3. 删除临时 DGDR，再创建最终 DGDR，使用这些名称设置覆盖配置。

性能分析开始后 DGDR spec 不可变，因此不能向临时资源添加或修改覆盖配置。覆盖配置只能修改生成的 DGD 中已有的组件，不能添加新的 worker、EPP 或其他拓扑组件。各后端和拓扑当前使用的 profiler 名称请参阅[生成的组件名称](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx#generated-component-names)。

> [!IMPORTANT]
> 旧版覆盖配置使用 `nvidia.com/v1alpha1` DGD 结构。为兼容性仍予以支持，但合并行为不同：
> `spec.services` 按服务名合并，worker 的 `extraPodSpec.mainContainer.args` 追加到生成的参数。
> 在 `v1beta1` 中，组件、容器和环境变量等 map list 按 `name` 合并；图级 `spec.env`
> 和容器 `args` 等 atomic list 则替换生成的列表。新覆盖配置请使用 `v1beta1`。
> 替换时提供完整的目标参数列表，或使用下面的追加修饰符。

例如，向生成的 frontend 参数追加 KV 路由参数：

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        components:
        - name: Frontend
          podTemplate:
            spec:
              containers:
              - name: main
                $patch:
                  args: append
                args:
                - --router-mode
                - kv
```

要向已生成显式参数的容器追加参数，在该具名容器上将 `$patch.args` 设为 `append`。覆盖辅助逻辑会在返回最终 DGD 前合并参数列表，并移除 `$patch` 指令。追加修饰符仅支持 `args: append`，要求参数列表非空；如果生成的 DGD 不包含目标容器，或目标未定义 `args`，则会失败。不使用 `$patch` 时，指定 `args` 仍会替换整个列表，需要提供完整的目标参数。

完整的合并、元数据和校验规则请参阅 [DGDR 参考：生成的 DGD 覆盖配置](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx#generated-dgd-overrides)。

### 性能分析 Job 覆盖配置与信任边界

`spec.overrides.profilingJob` 接受部分 Kubernetes `JobSpec`，Operator 将其合并到启动的性能分析 Job，例如添加容忍或调整资源。由于 Operator 代表用户创建 Job，应像对待其他 Kubernetes 工作负载创建 API 一样管理其权限：

> [!IMPORTANT]
> 在 Dynamo 运行的命名空间中，任何能创建工作负载的主体，无论直接创建 `Pod`，还是创建
> `Job`、`Deployment`、`DynamoGraphDeploymentRequest` 等会生成 Pod 的资源，都处于该命名空间
> 的信任边界内：它能运行代码，并使用该命名空间中挂载的 Secret 和 ServiceAccount token。
> 创建 DGDR（包括使用 `overrides.profilingJob`）是其中一种方式，其权限并不高于在同一
> 命名空间中创建 `Job` 或 `Pod`。这是预期设计，与 Kubernetes 对所有生成 Pod 的资源的处理一致。
>
> 通过 [Pod Security Admission](https://kubernetes.io/docs/concepts/security/pod-security-admission/)
> 及准入 webhook，**在最终生成的 Pod 上集中实施安全策略**，应用完整的
> [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)。
> 检查范围不仅包括 `securityContext`，还包括 `privileged`、宿主机命名空间、`hostPath` 卷和
> 宿主机端口。为命名空间添加标签前，PSA 不实施策略。请像管理其他工作负载一样，为所有运行
> Dynamo 的命名空间设置 `pod-security.kubernetes.io/enforce: <level>` 及对应的
> `audit`/`warn` 标签。Operator 不会重新实现这些检查。
>
> Dynamo Operator 生成的工作负载（包括 DGDR 性能分析 Job）满足 **`baseline`** 标准。
> 实施 **`baseline`** 可在保持 Dynamo 运行的同时，阻止特权容器、宿主机命名空间、宿主机设备
> 和 `hostPath` 带来的提权路径。
>
> PSA 管理 Pod 的安全配置，**不管理其身份**。`overrides.profilingJob` 可设置 Job 的
> `serviceAccountName` 和 `automountServiceAccountToken`，这些受 RBAC 和命名空间成员关系
> 约束，而非 PSA，权限与同一命名空间中的其他 Pod 创建者相同。因此，仅向可信任其在该命名空间
> 创建 Pod 的主体授予 DGDR 及其他工作负载资源的 `create`/`update` 权限，并以命名空间作为
> 租户边界。请参阅 [Kubernetes RBAC 最佳实践](https://kubernetes.io/docs/concepts/security/rbac-good-practices/#workload-creation)。

性能分析 Job 始终位于 DGDR 所在的命名空间，覆盖配置不能将其移到其他命名空间。但命名空间约束不等同于节点或跨租户隔离。如果准入策略允许特权容器、宿主机命名空间、宿主机设备或 `hostPath` 挂载，DGDR 创建者可以通过 Operator 获得这些能力，与同一命名空间中的其他 Pod 创建者相同。对每个生成的 Pod 强制执行 `baseline` 且不设置豁免，可阻断这些路径；需要更强隔离时，请使用命名空间作为租户边界。

## 后续步骤

| 目标 | 指南 |
|---|---|
| 可直接使用的 DGDR 清单 | [DGDR 模板](../../../../../pages/recipes/kubernetes-templates/dgdr.mdx)、[profiler 示例](../../../../../pages/developer-guide/knowledge-base/modular-components/profiler/profiler-examples.md) |
| 完整字段表和生命周期 | [DGDR 参考](../../../../../pages/reference/kubernetes-api/dynamo-graph-deployment-request.mdx) |
| 手动编写部署 | [DGD 指南](../../../../../pages/kubernetes/model-deployment/deploy-with-dgd.md) |
| 性能分析算法和模式 | [profiler 指南](../../../../../pages/developer-guide/knowledge-base/modular-components/profiler/profiler-guide.md) |
| 运行时自动扩缩容详情 | [Planner 指南](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md) |
| 完整 CRD 参考 | [API 参考](../../../../../pages/reference/kubernetes-api/full-api-reference.mdx) |
