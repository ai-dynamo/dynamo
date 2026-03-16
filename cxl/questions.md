# Dynamo 学习问题记录

这份文件用于持续记录学习过程中提出的问题，以及对应的详细解释。

记录原则：

- 不只记录“问题标题”
- 还要记录“为什么会困惑”
- 记录“源码里的真实调用链/数据流”
- 记录“阅读源码时应该怎么理解”

## 2026-03-16

### 问题 1

文件位置：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`

问题：

`func (r *DynamoGraphDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) ...`

这个函数最后是谁调用的？在仓库里没有直接搜到它的父调用函数。

详细回答：

这个 `Reconcile()` 不是被项目里某个普通“父函数”直接手动调用的，所以用 `rg "Reconcile("` 之类的方法，通常找不到一个清晰的业务调用栈。

它属于 Kubernetes operator 常见的“回调注册模型”：

1. 在 `deploy/operator/cmd/main.go` 里，operator 启动时会执行 `registerControllers(...)`。
2. `registerControllers(...)` 会创建 `DynamoGraphDeploymentReconciler`，然后调用它的 `SetupWithManager(mgr)`。
3. `SetupWithManager(mgr)` 里通过 controller-runtime 的 builder 把这个 reconciler 注册到 manager 上。
4. 随后 `mgr.Start(mainCtx)` 启动 controller manager。
5. 当被监听的资源发生事件时，controller-runtime 会自动构造 `ctrl.Request`，再回调 `Reconcile(ctx, req)`。

所以，这个函数真正的“调用者”不是你项目业务代码里的另一个函数，而是：

- controller-runtime 的 controller 事件循环

可以把这条链路记成：

```text
main()
  -> registerControllers()
  -> SetupWithManager(mgr)
  -> Complete(reconciler)
  -> mgr.Start()
  -> watch 到资源变化
  -> controller-runtime 调用 Reconcile(ctx, req)
```

为什么你搜不到“父函数”：

- 因为这不是静态直接调用
- 而是运行时注册后的事件驱动回调
- 类似 HTTP 框架里“你写 handler，框架来调用”

后续可继续追的源码入口：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`
- `deploy/operator/cmd/main.go`
- `deploy/operator/vendor/sigs.k8s.io/controller-runtime/pkg/builder/controller.go`

关键源码定位：

- `deploy/operator/cmd/main.go`
  这里有 `registerControllers(...)` 和 `mgr.Start(mainCtx)`
- `deploy/operator/internal/controller/dynamographdeployment_controller.go`
  这里有 `SetupWithManager(mgr)`
- `deploy/operator/vendor/sigs.k8s.io/controller-runtime/pkg/builder/controller.go`
  这里可以看到 `Complete()` / `Build()` 是如何把 reconciler 和 watch 绑定起来的

### 问题 2

文件位置：

- `deploy/operator/internal/controller/dynamographdeployment_controller.go`

问题：

请仔细讲一下 `SetupWithManager()` 这个函数。

关注点：

- 它到底注册了什么
- `For / Owns / Watches / Complete` 分别是什么意思
- 为什么 `Reconcile()` 不是直接调用，而是通过这里接入 controller-runtime

详细回答：

`SetupWithManager()` 的作用，不是执行业务 reconcile，而是把这个 controller 的“监听规则、事件来源、事件过滤方式、最终回调对象”注册给 controller-runtime。

一句话理解：

- `Reconcile()` 是“出事之后怎么处理”
- `SetupWithManager()` 是“什么事算出事、谁来通知我、通知后交给谁处理”

这个函数的主骨架大致是：

```
func (r *DynamoGraphDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&DynamoGraphDeployment{}, ...).
		Named(...).
		Owns(&DynamoComponentDeployment{}, ...).
		Owns(&DynamoGraphDeploymentScalingAdapter{}, ...).
		Owns(&PersistentVolumeClaim{}, ...).
		WithEventFilter(...)

	if r.RuntimeConfig.GroveEnabled {
		ctrlBuilder = ctrlBuilder.Owns(&PodCliqueSet{}, ...).
			Watches(&PodClique{}, ...)
	}

	observedReconciler := observability.NewObservedReconciler(r, ...)
	return ctrlBuilder.Complete(observedReconciler)
}
```

可以分成 7 个层次理解。

#### 1. `NewControllerManagedBy(mgr)`：在 manager 下面创建一个 controller builder

这里的 `mgr` 是 operator 总控，来自 `deploy/operator/cmd/main.go`。

它负责：

- 提供 k8s client
- 提供 cache / informer
- 管理 controller 生命周期
- 提供健康检查、webhook、metrics 等基础设施

所以这行的意思不是“开始监听了”，而是：

- 我要定义一个新的 controller
- 这个 controller 归这个 manager 管

#### 2. `For(...)`：声明主资源

```go
For(&nvidiacomv1alpha1.DynamoGraphDeployment{}, ...)
```

表示：

- 这个 controller 的主资源是 `DynamoGraphDeployment`

主资源的含义是：

- 这个 controller 最终要 reconcile 的核心对象是谁
- 当它发生符合条件的变化时，生成的 request 会直接指向这个对象

这里还加了：

```go
builder.WithPredicates(predicate.GenerationChangedPredicate{})
```

这表示：

- 不是所有 update 都触发
- 只有 generation 变化时才触发

在 Kubernetes 里，generation 一般代表：

- 用户期望状态（通常是 spec）发生了变化

所以这一步可以理解成：

- “我关心 DGD”
- “但主要关心 spec 变化，不想被 status 更新反复打扰”

#### 3. `Named(...)`：给 controller 命名

```go
Named(consts.ResourceTypeDynamoGraphDeployment)
```

这主要用于：

- 日志
- metrics
- 调试输出

它不改变业务逻辑，但能让运行时更容易区分不同 controller。

#### 4. `Owns(...)`：监听自己创建/拥有的子资源

这里有几类：

- `DynamoComponentDeployment`
- `DynamoGraphDeploymentScalingAdapter`
- `PersistentVolumeClaim`
- Grove 开启时还有 `PodCliqueSet`

`Owns(...)` 的语义不是“我在代码里会用到这个类型”，而是：

- 如果这些下游资源发生变化，也应该回头触发它们所属 DGD 的 reconcile

这是 operator 的典型模式：

- DGD 是主资源
- DCD / PVC / ScalingAdapter / PodCliqueSet 是它派生出来的子资源
- 子资源状态变化会影响 DGD 是否 ready，或者是否要继续推进下一步

所以 `Owns(...)` 本质是在建立：

- “子资源事件 -> 主资源 request”

##### 为什么这里还给 `Owns(...)` 配 predicate

因为 controller 不希望所有子资源事件都反复触发 reconcile。

例如有的地方会显式忽略 create 事件：

- 因为资源就是当前 controller 自己刚创建的
- 如果创建后立刻又因为 create 事件再触发一轮，有时只是在制造噪音

所以这些 predicate 是在做事件降噪。

#### 5. `Watches(...)`：自定义 watch 映射关系

在 Grove 打开时，代码里还有：

```go
Watches(
  &grovev1alpha1.PodClique{},
  handler.EnqueueRequestsFromMapFunc(r.mapPodCliqueToRequests),
  ...
)
```

这和 `Owns(...)` 不完全一样。

`Owns(...)` 更像：

- “标准 ownerReference 父子关系的监听”

而 `Watches(...)` 更像：

- “我自己指定一种资源，再自己写规则，把它映射成某个 request”

这里的意思是：

- 监听 `PodClique`
- 当它发生我们关心的状态变化时
- 调用 `mapPodCliqueToRequests(...)`
- 把这个 `PodClique` 映射回所属的 `DynamoGraphDeployment`

为什么 Grove 路径要专门 watch `PodClique`：

- 因为 Grove 资源的真实 readiness 变化，经常体现在 `PodClique`
- controller 需要根据 `PodClique` 的状态变化，重新评估整个 DGD 是否 ready

#### 6. `WithEventFilter(...)`：给整个 controller 再加一层总过滤器

```go
WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(...))
```

前面的 `For / Owns / Watches` 解决的是：

- 监听哪些资源

这里的 `WithEventFilter(...)` 更像是：

- 对整个 controller 的事件再做一层总开关/总过滤

所以你可以把它理解成两层筛选：

1. 资源维度：哪些资源值得监听
2. 事件维度：监听到的事件里，哪些值得真正入队

#### 7. `Complete(observedReconciler)`：注册完成

这一步最关键，但最容易被低估。

代码：

```go
observedReconciler := observability.NewObservedReconciler(r, ...)
return ctrlBuilder.Complete(observedReconciler)
```

这里不是直接 `Complete(r)`，而是先包了一层 `observedReconciler`。

这说明：

- 真正的业务 reconciler 还是 `r`
- 但外面再包了一层观测逻辑，用于 metrics、日志、耗时统计之类的横切能力

而 `Complete()` 在 controller-runtime 里的工作，大体是：

- `Build(r)`
- 绑定 controller 和 reconciler
- 把前面声明的 `For / Owns / Watches` 都注册成 watch

所以 `Complete(...)` 才是：

- “从 builder 配置阶段”
- “进入真正 controller 构建阶段”

#### 为什么 `Reconcile()` 不直接手动调用

因为 operator 是事件驱动系统，不是普通函数调用系统。

你不应该把它想成：

```go
func main() {
    Reconcile(dgd)
}
```

而应该想成：

```text
先声明：
  我关心哪些资源
  资源变化后怎么映射 request
  request 到来后交给哪个 reconciler

再由 controller-runtime 负责：
  建 informer
  收事件
  入队
  去重
  重试
  并发执行
  最终回调 Reconcile(ctx, req)
```

所以 `SetupWithManager()` 的真实角色是：

- 把你的业务处理器 `Reconcile`
- 接进 controller-runtime 的事件处理框架里

#### 一张完整心智图

```text
main.go
  -> registerControllers()
  -> reconciler.SetupWithManager(mgr)
     -> For(DGD)
     -> Owns(DCD / PVC / ScalingAdapter / PodCliqueSet)
     -> Watches(PodClique -> map to DGD)
     -> Complete(observedReconciler)
  -> mgr.Start()
  -> 资源事件进入 workqueue
  -> controller-runtime 调用 Reconcile(ctx, req)
```

#### 阅读这个函数时最重要的三个问题

以后你看任何一个 `SetupWithManager()`，都可以固定问自己三件事：

1. 这个 controller 的主资源是谁？
2. 它还监听了哪些下游资源？
3. 哪些事件会真正触发 reconcile？

只要这三件事答清楚，这个 controller 的“事件入口模型”你就掌握了。
