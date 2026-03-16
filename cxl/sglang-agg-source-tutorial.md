# 从 `agg.yaml` 开始学习 Dynamo：SGLang 聚合部署源码教程

这篇教程的目标不是“教你把 YAML 跑起来”这么简单，而是带你顺着一条最真实的主线，从 Kubernetes 部署入口一路走到 Dynamo 的核心运行时。

我们选择的入口文件是：

- `examples/backends/sglang/deploy/agg.yaml`

这条链路非常适合第一次系统阅读 Dynamo：

1. 先看一个最小可运行的 Kubernetes 部署描述。
2. 再看它如何被 Operator 解释成多个组件。
3. 再看每个组件最终启动了什么进程。
4. 最后看请求如何从 Frontend 流到 SGLang worker。

---

## 一、先建立一张总图

`agg.yaml` 部署的是一个 `DynamoGraphDeployment`，它描述了一张“推理图”。

在这个聚合示例里，图里只有两个服务：

- `Frontend`：对外提供 OpenAI 兼容 HTTP API。
- `decode`：SGLang worker，负责真正加载模型并生成结果。

你可以先把它理解成：

```text
Client
  -> Frontend
  -> Router / Preprocess
  -> decode worker (SGLang)
  -> Frontend stream back
  -> Client
```

在 Kubernetes 里，它不会直接变成 Pod，而是先经过 Operator，再根据编排路径落到不同的底层资源：

```text
agg.yaml
  -> DynamoGraphDeployment (DGD)
  -> Operator Reconcile
  -> 选择编排路径
     -> Grove 路径
        -> PodCliqueSet / PodClique / PodCliqueScalingGroup
        -> 可选注入 kai-scheduler
        -> Pod
     -> Component 路径
        -> DynamoComponentDeployment (DCD)
        -> Deployment / Service / Pod
        -> 多机时可走 LeaderWorkerSet，并可配合 Volcano
  -> 容器内启动 dynamo.frontend / dynamo.sglang
```

这个项目里很关键的一点是：

- `DGD` 负责描述“整张图”。
- `DCD` 负责描述“图中的单个组件”。
- Operator 负责把“图”拆成“组件”，并选择底层编排器。
- Grove 是 Dynamo 在 Kubernetes 上的一条重要编排路径，不只是外围项目。
- 调度器不是同一个层面的概念，它通常在 Pod 模板生成后参与最终 placement。

这里要特别纠正我前一版里省略掉的部分：

- Grove 不是可有可无的“额外项目”，而是 Dynamo Operator 可能直接生成的目标资源类型。
- `agg.yaml` 虽然是单机聚合例子，但只要 Operator 运行时判定走 Grove 路径，它也可能被编排成 Grove 的 `PodCliqueSet`，而不一定是 `DCD -> Deployment`。
- `DCD -> Deployment` 更准确地说是 “非 Grove 的组件路径”。

所以更完整的理解应该是：

```text
DGD 负责声明目标拓扑
Operator 负责选编排路径
Grove / DCD / LWS 负责把组件组织成 K8s 工作负载
Volcano / Kai-Scheduler 负责调度这些工作负载到合适节点
```

---

## 二、第一章：先读懂 `agg.yaml`

源码入口：

- `examples/backends/sglang/deploy/agg.yaml`

这个文件本质上声明了一个 CR：

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: sglang-agg
```

这说明它不是标准 Kubernetes 内建对象，而是 Dynamo Operator 提供的自定义资源。

### 1. `spec.services`

最重要的字段是：

```yaml
spec:
  services:
    Frontend:
      componentType: frontend
    decode:
      componentType: worker
```

这表示这张图里有两个服务节点。

- `Frontend` 的类型是 `frontend`
- `decode` 的类型是 `worker`

这里的 key 不是随便写的，它后面会成为：

- 服务名
- DCD 名称的一部分
- Dynamo 内部 namespace/component 路径的一部分

### 2. `replicas`

```yaml
replicas: 1
```

表示该服务期望的 Pod 副本数。

在这个示例里：

- Frontend 1 个副本
- decode 1 个副本

### 3. `extraPodSpec.mainContainer`

`agg.yaml` 最值得注意的是它把默认容器配置做了覆盖：

```yaml
extraPodSpec:
  mainContainer:
    image: my-registry/sglang-runtime:my-tag
```

对于 `decode` 还继续覆盖了：

```yaml
command:
  - python3
  - -m
  - dynamo.sglang
args:
  - --model-path
  - Qwen/Qwen3-0.6B
```

这告诉你两个事实：

1. Operator 会先生成一套“默认 Pod 模板”。
2. 用户 YAML 可以用 `extraPodSpec.mainContainer` 精准覆写容器镜像、工作目录、命令和参数。

### 4. `envFromSecret`

```yaml
envFromSecret: hf-token-secret
```

这个 secret 主要给 worker 下载 HuggingFace 模型用。

### 5. `resources`

```yaml
resources:
  limits:
    gpu: "1"
```

表示 decode worker 需要 1 张 GPU。

在学习上很重要的一点是：

- Frontend 通常是 CPU 服务。
- Worker 才是真正消耗 GPU 的地方。

---

## 三、第二章：这个 YAML 为什么能被 `kubectl apply`？

先看 CRD 定义相关源码：

- `deploy/operator/api/v1alpha1/dynamographdeployment_types.go`
- `deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml`

`DynamoGraphDeploymentSpec` 里最核心的字段就是：

- `Services map[string]*DynamoComponentDeploymentSharedSpec`
- `Envs`
- `BackendFramework`
- `Restart`

这说明 `agg.yaml` 里 `spec.services.Frontend` 和 `spec.services.decode` 的每一项，最终都被解析为一个 `DynamoComponentDeploymentSharedSpec`。

也就是说：

- DGD 的 `services` 不是直接生成 Pod
- 而是先复用一套“组件级别”的公共 spec 结构

这也是为什么 Dynamo 的图部署和单组件部署能共用大量逻辑。

---

## 四、第三章：Operator 如何接住这个 DGD

推荐源码：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`
- `docs/kubernetes/dynamo-operator.md`

最关键的方法是：

- `DynamoGraphDeploymentReconciler.Reconcile`

你可以把它理解成：

1. 读取当前 namespace/name 对应的 DGD。
2. 根据 spec 生成目标资源。
3. 把目标资源和集群现状做对齐。
4. 更新 DGD 的 status。

这段 controller 的主逻辑可以重点看三个问题。

### 0. 先决定走哪条编排路径

`dynamographdeployment_controller.go` 里有个很关键的方法：

- `isGrovePathway`

它的逻辑大意是：

- 如果 Grove 在当前集群可用，并且 DGD 没有显式加 `nvidia.com/enable-grove=false`
- 那么优先走 Grove 路径
- 否则走普通组件路径

所以从 controller 角度，真实主线不是一条，而是：

```text
DGD
  -> Reconcile
  -> isGrovePathway?
     -> yes: reconcileGroveResources
     -> no:  reconcileDynamoComponentsDeployments
```

### 1. `reconcileResources`

它会先处理图级别资源，比如：

- PVC
- checkpoint
- scaling adapter
- k8s discovery 相关资源
- EPP 相关资源

### 2. `reconcileGroveResources` 或 `reconcileDynamoComponentsDeployments`

这是本教程里最重要的一组分叉。

- Grove 路径：生成 `PodCliqueSet`，并由 Grove operator 继续展开成 `PodClique` / `PodCliqueScalingGroup`
- Component 路径：生成多个 `DCD`

也就是说，DGD 并不总是先变成 DCD。

---

## 四点五、补上你提到的 Grove、Volcano、Kai-Scheduler

这一层最容易混淆，因为“编排器”和“调度器”不是一个概念。

### 1. Grove 在哪里参与

推荐源码：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`
- `deploy/operator/internal/dynamo/graph.go`
- `deploy/operator/internal/dynamo/grove.go`

当 controller 走 Grove 路径时，会调用：

- `reconcileGroveResources`
- `GenerateGrovePodCliqueSet`

这时 Operator 生成的就不是 DCD，而是：

- `PodCliqueSet`

随后 Grove operator 再根据这个 `PodCliqueSet` 去管理：

- `PodClique`
- `PodCliqueScalingGroup`

所以 Grove 是“底层工作负载编排器”，不是纯粹文档里顺手提一下的依赖。

### 2. LeaderWorkerSet 在哪里参与

推荐源码：

- `deploy/operator/internal/controller/dynamocomponentdeployment_controller.go`
- `deploy/operator/internal/dynamo/lws.go`

如果不走 Grove，但服务是多机部署，DCD controller 会优先尝试：

- `LeaderWorkerSet`

这条路径主要是 Dynamo 的“非 Grove 多机编排方案”。

### 3. Volcano 在哪里参与

Volcano 主要出现在非 Grove 的多机组件路径里。

推荐源码：

- `deploy/operator/internal/controller/dynamocomponentdeployment_controller.go`

当 DCD 走 `LeaderWorkerSet` 路径时，controller 会为每个实例生成：

- `PodGroup`

这是 Volcano 的 gang scheduling 资源，用来保证一组相关 Pod 尽量整体调度。

所以你可以理解为：

- LWS 负责 leader/worker 结构
- Volcano 负责 gang scheduling

### 4. Kai-Scheduler 在哪里参与

推荐源码：

- `deploy/operator/internal/dynamo/grove.go`
- `deploy/helm/charts/platform/values.yaml`

Kai-Scheduler 主要和 Grove 路径集成。

在 `GenerateGrovePodCliqueSet` 里，如果：

- Grove 已启用
- Kai-Scheduler 已启用

那么 Operator 会在生成的 Grove clique 上自动注入：

- `schedulerName: kai-scheduler`
- `kai.scheduler/queue` label

也就是说：

- Grove 负责组织 cliques/scaling groups
- Kai-Scheduler 负责更智能地把这些 Pod 放到具体节点

这两者经常一起出现，但职责不同。

### 5. 所以完整链路应该怎么记

你可以把 Kubernetes 这层拆成两层：

```text
编排层:
  Operator -> Grove 或 DCD/LWS

调度层:
  kube-scheduler / Volcano / kai-scheduler
```

如果进一步代入 Dynamo 的常见路径，可以记成：

```text
路径 A: DGD -> Grove -> PodCliqueSet/PodCliqueScalingGroup -> kai-scheduler -> Pod
路径 B: DGD -> DCD -> Deployment -> kube-scheduler -> Pod
路径 C: DGD -> DCD -> LeaderWorkerSet + PodGroup -> Volcano -> Pod
```

### 6. Operator 怎么知道这些能力是否存在

推荐源码：

- `deploy/operator/api/config/v1alpha1/types.go`
- `deploy/operator/internal/controller_common/runtime.go`
- `deploy/operator/internal/controller_common/predicate.go`

这里还有一个容易忽略但非常关键的点：

- Grove/LWS/Kai-Scheduler 是否启用，并不只是看 YAML
- Operator 启动后会把“配置覆盖”和“集群自动探测”合并成运行时状态

相关运行时结构是：

- `RuntimeConfig`

里面有：

- `GroveEnabled`
- `LWSEnabled`
- `KaiSchedulerEnabled`

而自动探测逻辑会去检查对应 API group 是否已经注册到集群里，比如：

- `grove.io`
- `leaderworkerset.x-k8s.io`
- `scheduling.volcano.sh`
- `scheduling.run.ai`

这意味着从源码阅读角度，你可以这样理解：

```text
静态配置层:
  OperatorConfiguration.Orchestrators

运行时能力层:
  RuntimeConfig

控制器决策层:
  isGrovePathway / reconcileLeaderWorkerSetResources / injectKaiSchedulerIfEnabled
```

也就是说，Operator 不是“盲目假设集群有 Grove/Volcano/Kai-Scheduler”，而是会先探测，再决定能不能走那条路径。

### 7. 为什么要把“编排器”和“调度器”分开记

这个项目最容易让人脑子打结的地方，就是下面几个名字会一起出现：

- Grove
- LeaderWorkerSet
- Volcano
- Kai-Scheduler

但它们在语义上不是同一类东西。

#### 编排器更像“怎么组织一组 Pod”

例如：

- Grove 把一组相关 Pod 组织成 `PodCliqueSet / PodClique / PodCliqueScalingGroup`
- LWS 把一组相关 Pod 组织成 `LeaderWorkerSet`

它们重点解决的是：

- 哪些 Pod 属于同一个分布式组件
- leader 和 worker 的关系是什么
- 组件扩缩容时要把哪些 Pod 当成一个整体

#### 调度器更像“把这些 Pod 放到哪里”

例如：

- 默认 `kube-scheduler`
- `Volcano`
- `kai-scheduler`

它们重点解决的是：

- 这些 Pod 最终落到哪些节点
- 是否需要 gang scheduling
- 是否需要按 queue / topology / 资源约束做更智能放置

如果你把这两层混在一起，就会误以为：

- “用了 Grove 就等于用了 Kai-Scheduler”
- “用了 LWS 就等于一定是 Volcano”

但源码上并不是这个关系。

更准确地说：

- Grove 可以不配 Kai-Scheduler
- LWS 路径强调的是 `LeaderWorkerSet`，而 Volcano 是它常见的 gang scheduling 配套
- 单机 `Deployment` 路径可能完全只用默认 `kube-scheduler`

### 8. 结合 `agg.yaml` 应该怎么落地理解

回到你最初关心的 `agg.yaml`，这一份 YAML 最适合被当成：

- “理解 Dynamo 控制面与运行时主链路的最小入口”

但它不是：

- “整个 Kubernetes 编排分支的完整代表”

更准确地说：

- 它让你最容易先看懂 `Frontend + 单 worker` 这条最短业务链
- 但在 Operator 层，背后仍然有 Grove 路径、DCD 路径、LWS 路径这些分支逻辑
- 在调度层，又可能叠加默认调度器、Volcano、Kai-Scheduler

所以你后面学习时最好始终把问题拆成两句：

1. 这个组件是被谁编排出来的？
2. 这些 Pod 最终由谁调度到节点上？

---

## 五、第四章：DGD 如何拆成多个 DCD

推荐源码：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`
- `deploy/operator/internal/dynamo/graph.go`

关键函数：

- `reconcileDynamoComponentsDeployments`
- `GenerateDynamoComponentsDeployments`
- `generateSingleDCD`

这一章要带一个前提去看：

- 只有在“不走 Grove 的组件路径”下，DGD 才会先被拆成多个 DCD

`GenerateDynamoComponentsDeployments` 会遍历：

- `parentDGD.Spec.Services`

对 `Frontend`
和 `decode`
分别生成一个 `DynamoComponentDeployment`。

### 这里最值得你观察的几个点

#### 1. DCD 名字怎么来

`generateSingleDCD` 里会调用：

- `GetDCDResourceName`

于是这两个服务大致会变成：

- `sglang-agg-frontend`
- `sglang-agg-decode`

如果是 worker rolling update，还会带 hash 后缀。

#### 2. service key 如何进入运行时命名空间

DGD 会根据 namespace + graph name 计算出 Dynamo 内部 namespace。

随后每个 DCD 都会写入：

- `Spec.ServiceName`
- `Spec.DynamoNamespace`

这决定了后面 frontend 和 worker 在 Dynamo runtime 里彼此发现时使用的逻辑路径。

#### 3. 图级 env 如何下沉到组件

`GenerateDynamoComponentsDeployments` 里会把 DGD 顶层的 `spec.envs` 合并到每个 DCD。

也就是说：

- 图级配置可以做全局默认值
- 组件级配置可以做局部覆盖

#### 4. `extraPodSpec` 不会丢

`agg.yaml` 里写的 `extraPodSpec.mainContainer.image/command/args`
会原封不动地下沉到 DCD，后面继续参与 Pod 模板生成。

---

## 六、第五章：DCD 如何继续变成 Deployment / Service / Pod

推荐源码：

- `deploy/operator/internal/controller/dynamocomponentdeployment_controller.go`
- `deploy/operator/api/v1alpha1/dynamocomponentdeployment_types.go`

关键方法：

- `DynamoComponentDeploymentReconciler.Reconcile`
- `reconcileDeploymentResources`
- `generateDeployment`
- `generatePodTemplateSpec`

这一层你可以理解成：

```text
DCD = Dynamo 自己的组件抽象
Deployment/Service = Kubernetes 原生资源
```

Controller 做的事情是：

1. 为这个组件生成 Deployment。
2. 为这个组件生成 Service。
3. 必要时生成 Ingress / VirtualService。
4. 更新 DCD status。

### `generatePodTemplateSpec` 是你要精读的函数

这里几乎决定了 Pod 最终长什么样。

它内部会调用：

- `GenerateBasePodSpecForController`

然后再把：

- labels
- annotations
- serviceAccount
- checkpoint
- metrics label

这些控制器层面的东西补进去。

---

## 七、第六章：默认容器配置是怎么来的

推荐源码：

- `deploy/operator/internal/dynamo/component_common.go`
- `deploy/operator/internal/dynamo/component_frontend.go`
- `deploy/operator/internal/dynamo/component_worker.go`
- `deploy/operator/internal/dynamo/graph.go`

这一层很关键，因为它决定了“如果你没写 `extraPodSpec`，Operator 默认给你什么命令和环境变量”。

### 1. 公共默认值

`component_common.go` 里 `BaseComponentDefaults` 会先提供公共环境变量，比如：

- `DYN_NAMESPACE`
- `DYN_COMPONENT`
- `POD_NAME`
- `POD_NAMESPACE`
- `POD_UID`

如果 discovery backend 不是 etcd，还会注入：

- `DYN_DISCOVERY_BACKEND=kubernetes`

这一步很重要，因为它解释了为什么 worker 能用 Kubernetes CR 做服务发现。

### 2. frontend 默认值

`component_frontend.go` 里默认主容器会被设置成：

```text
python3 -m dynamo.frontend
```

同时补上：

- HTTP 端口
- `/live`
- `/health`
- `DYN_HTTP_PORT`

### 3. worker 默认值

`component_worker.go` 里 worker 会默认补上：

- `DYN_SYSTEM_ENABLED=true`
- `DYN_SYSTEM_PORT`
- NIXL telemetry 相关环境变量
- `/live` `/health` `/startup` probes

### 4. 用户 YAML 如何覆盖默认值

在 `agg.yaml` 里，`decode` 明确把命令改成了：

```text
python3 -m dynamo.sglang ...
```

所以最终 Pod 并不是单纯使用 controller 的默认值，而是：

```text
默认模板 + backend 补丁 + extraPodSpec 覆盖
```

这是 Dynamo Operator 很重要的设计风格。

---

## 八、第七章：SGLang backend 在 Operator 里做了什么

推荐源码：

- `deploy/operator/internal/dynamo/backend_sglang.go`

这个文件不是启动 SGLang 本身，而是“在生成 PodSpec 时，为 SGLang 组件补充 backend 特有逻辑”。

当前你在 `agg.yaml` 这个单机聚合场景里，`backend_sglang.go` 做的事情并不多。

因为：

- 单节点时 `UpdateContainer` 基本直接返回
- 多节点时才会自动注入 `--dist-init-addr --nnodes --node-rank`

所以这里顺便能理解一个设计分层：

- Operator 层负责组装 Pod 配置
- runtime 层负责真正运行 frontend / worker
- backend adapter 层负责插入框架特有参数

---

## 九、第八章：容器启动后，Frontend 进程做了什么

推荐源码：

- `components/src/dynamo/frontend/__main__.py`
- `components/src/dynamo/frontend/main.py`

入口很简单：

- `python -m dynamo.frontend`
- `__main__.py` 调 `main.py`

真正重要的是 `async_main()`。

它大概做这几件事：

1. 解析 frontend CLI 参数。
2. 构造 `DistributedRuntime`。
3. 选择 router mode。
4. 创建 engine / processor。
5. 启动 HTTP server。
6. 通过 discovery 自动发现 worker。

你要重点关注两个概念。

### 1. `DistributedRuntime`

这是 Dynamo 运行时抽象。

Frontend 和 worker 并不是靠 K8s Service 名称直接硬编码互连，而是先注册到 Dynamo runtime 的发现系统里，再由 runtime 建立路由关系。

### 2. RouterConfig

在 `agg.yaml` 里虽然没有显式开启 `kv` router，但 frontend 本质上仍然承担：

- HTTP API 接入
- 请求预处理
- 选择 worker
- 把结果流式返回

在更复杂场景里，只是 router 策略会变得更强。

---

## 十、第九章：容器启动后，SGLang worker 做了什么

推荐源码：

- `components/src/dynamo/sglang/__main__.py`
- `components/src/dynamo/sglang/main.py`
- `components/src/dynamo/sglang/init_llm.py`

`agg.yaml` 中 decode worker 的命令是：

```text
python3 -m dynamo.sglang --model-path ... --served-model-name ...
```

其主线是：

1. 解析 SGLang + Dynamo 参数。
2. 创建 `DistributedRuntime`。
3. 构造 SGLang engine。
4. 创建 Dynamo endpoint。
5. 调用 `serve_endpoint()` 对外提供生成能力。
6. 调用 `register_model_with_readiness_gate()` 把自己注册进 discovery。

### 为什么 `decode` 叫 worker，但还能被 frontend 找到？

因为在 `init_decode()` 里它做了两件事：

1. `runtime.endpoint(...)`
2. `register_model_with_readiness_gate(...)`

也就是说它不只是“起了一个 Python 进程”，而是主动向 Dynamo runtime 报告：

- 我是谁
- 我提供哪个 endpoint
- 我服务哪个模型
- 我当前是否 ready

Frontend 随后才能把请求路由给它。

---

## 十一、第十章：Kubernetes discovery 到底是怎么工作的

推荐源码：

- `lib/runtime/src/discovery/kube/crd.rs`
- `lib/runtime/src/discovery/metadata.rs`

这一章非常关键，因为它把“Dynamo 内部发现”跟“Kubernetes 对象”接起来了。

在 Kubernetes discovery 模式下，worker 会把自己的 discovery metadata 写成一个 CR：

- `DynamoWorkerMetadata`

而且这个 CR 会：

- 以 Pod 名命名
- ownerReference 指向 Pod

所以：

- Pod 创建后，worker 启动并注册 metadata
- Pod 删除后，这个 metadata CR 也会被 Kubernetes 自动垃圾回收

`DiscoveryMetadata` 里维护的是：

- endpoints
- model_cards
- event_channels

Frontend 做发现时，本质上是在查这些注册信息。

这就解释了为什么 Dynamo 不需要把所有 worker 地址手写到 frontend 配置里。

---

## 十二、第十一章：一条请求是怎么走完的

在 `agg.yaml` 这个聚合场景里，你可以先用下面这条最简主线理解：

1. 用户请求打到 Frontend 的 HTTP API。
2. Frontend 完成 OpenAI 协议解析、预处理、路由选择。
3. Frontend 通过 `DistributedRuntime` 找到已注册的 `decode` worker endpoint。
4. SGLang worker 收到请求，调用底层 SGLang engine 推理。
5. 生成结果以流式或非流式方式返回给 Frontend。
6. Frontend 再按 OpenAI 兼容格式返回给客户端。

如果你以后继续学习 `disagg.yaml`，这条链路会进一步拆成：

- prefill worker
- decode worker
- 可能还有 KV router / planner

但 `agg.yaml` 的好处就在于它只有一条最短路径，特别适合第一次建立全局认知。

---

## 十三、建议你按这个顺序读源码

这是我最推荐的章节式学习顺序。

### 第 1 篇：部署入口长什么样

先读：

- `examples/backends/sglang/deploy/agg.yaml`
- `examples/backends/sglang/deploy/README.md`

目标：

- 看懂有哪些组件
- 看懂哪些字段会影响容器启动

### 第 2 篇：DGD 是什么

再读：

- `deploy/operator/api/v1alpha1/dynamographdeployment_types.go`

目标：

- 看懂 `services` 结构
- 看懂为什么它是“图级 CR”

### 第 3 篇：Operator 如何处理 DGD

再读：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`

目标：

- 只盯住 `Reconcile`
- 重点盯住 `reconcileDynamoComponentsDeployments`

### 第 4 篇：图如何拆成组件

再读：

- `deploy/operator/internal/dynamo/graph.go`

目标：

- 重点看 `GenerateDynamoComponentsDeployments`
- 弄清 DCD 名字、env 合并、replicas、restart 注解

### 第 5 篇：组件如何落成 Deployment

再读：

- `deploy/operator/internal/controller/dynamocomponentdeployment_controller.go`

目标：

- 重点看 `generateDeployment`
- 重点看 `generatePodTemplateSpec`

### 第 6 篇：默认容器配置怎么拼

再读：

- `deploy/operator/internal/dynamo/component_common.go`
- `deploy/operator/internal/dynamo/component_frontend.go`
- `deploy/operator/internal/dynamo/component_worker.go`
- `deploy/operator/internal/dynamo/backend_sglang.go`

目标：

- 看清默认 command/args/env/probe 怎么来
- 看清 backend-specific 补丁在哪里加

### 第 7 篇：Frontend 进程做什么

再读：

- `components/src/dynamo/frontend/main.py`

目标：

- 看清 runtime 初始化
- 看清 router 初始化
- 看清 HTTP 服务入口

### 第 8 篇：SGLang worker 进程做什么

再读：

- `components/src/dynamo/sglang/main.py`
- `components/src/dynamo/sglang/init_llm.py`

目标：

- 看清 engine 初始化
- 看清 endpoint 注册
- 看清 readiness 注册

### 第 9 篇：K8s discovery 怎么把 frontend 和 worker 连起来

再读：

- `lib/runtime/src/discovery/kube/crd.rs`
- `lib/runtime/src/discovery/metadata.rs`

目标：

- 看清 `DynamoWorkerMetadata`
- 看清发现信息的数据结构

---

## 十四、每一篇你都可以问自己的问题

为了避免“代码看完了，但脑子里没有模型”，建议你每一篇都只回答 3 个问题：

1. 这一层输入是什么？
2. 这一层输出是什么？
3. 这一层把上游信息增加了什么？

比如：

- DGD controller 的输入是 DGD，输出是 DCD
- DCD controller 的输入是 DCD，输出是 Deployment/Service
- worker runtime 的输入是容器参数，输出是可发现的 endpoint

只要这三个问题能答出来，你就不会迷路。

---

## 十五、最适合你的下一步

如果你准备“从前到后”慢慢吃透这个项目，我建议下一轮就从下面这个顺序开始：

1. 先逐行讲 `agg.yaml`
2. 再逐行讲 `dynamographdeployment_types.go`
3. 再逐行讲 `dynamographdeployment_controller.go` 里的主流程

这样你的第一阶段会先把控制面吃透；等控制面清楚了，再下潜到 frontend 和 sglang runtime，会轻松很多。

如果你愿意，我们下一篇我可以直接按这份教程的第 1 篇开始，带你逐行拆 `agg.yaml`，并且把每个字段对应到后续源码位置。
