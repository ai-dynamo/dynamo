# Dynamo 学习问题记录（续）

说明：

- 当前工作区在 `2026-03-30` 时不存在之前对话里使用过的 `cxl/questions.md` / `cxl/questions-2.md` 历史文件。
- 这个文件是在当前 worktree 里重新建立的续写版本。
- 后续我会继续把新的问题和回答记录到这里；如果超过 3000 行，再切到 `questions-3.md`。

## 2026-03-30

### 问题 1

问题：

继续讲 `make_engine` 在 Rust 里到底做了什么，以及它怎么把 Python 的 `chat_engine_factory` 接进 Rust runtime。

详细回答：

这个问题正好是你从 Python 代码进入 Rust runtime 的关键桥。

如果先用一句话总结：

**`make_engine` 的作用，就是把 Python 侧传进来的 `EntrypointArgs`、`DistributedRuntime`、`chat_engine_factory` 等信息，组装成 Rust 侧真正能跑的 engine 配置；其中 `chat_engine_factory` 会被包装成一个 Rust 可回调的异步 callback。**

你可以把这条链先记成：

```text
frontend/main.py
  -> EntrypointArgs(...)
  -> make_engine(runtime, e)     # Python 调 Rust
  -> Rust 组装 EngineConfig
  -> Rust 保存/包装 chat_engine_factory
  -> 运行时发现模型实例
  -> Rust 再回调 Python 的 chat_engine_factory
  -> Python 返回 PythonAsyncEngine
  -> Rust 持有这个 engine 并驱动 HTTP 请求流
```

所以它不是单向调用，而是一条：

- Python -> Rust
- Rust -> Python

的双向桥接链路。

#### 一、先看 `make_engine` 的输入是什么

在 `frontend/main.py` 里你看到的是：

```python
e = EntrypointArgs(EngineType.Dynamic, **kwargs)
engine = await make_engine(runtime, e)
```

这里的两个输入分别是：

1. `runtime`
    - Python 侧的 `DistributedRuntime`
    - 其实本质上也是 Rust 绑定出来的对象
2. `e`
    - `EntrypointArgs`
    - 里面装着 frontend 的各种配置

在 Rust 里，`EntrypointArgs` 定义在：

- `lib/bindings/python/rust/llm/entrypoint.rs`

它里面包含这些关键字段：

- `engine_type`
- `model_path`
- `model_name`
- `endpoint_id`
- `context_length`
- `template_file`
- `router_config`
- `kv_cache_block_size`
- `http_host`
- `http_port`
- `tls_cert_path`
- `tls_key_path`
- `runtime_config`
- `namespace`
- `namespace_prefix`
- `migration_limit`
- `chat_engine_factory`

所以你可以把 `EntrypointArgs` 理解成：

- frontend 对“我要启动一个怎样的请求处理 engine”的完整声明

#### 二、`EntrypointArgs` 在 Rust 里先做了一件很重要的事：捕获 `chat_engine_factory` 的 TaskLocals

这个点很关键。

在 `EntrypointArgs::new(...)` 里，Rust 会处理：

```rust
chat_engine_factory: Option<PyObject>
```

并把它变成：

- `Option<PyEngineFactory>`

里面除了 callback 本身，还会保存：

- `TaskLocals`

代码的真实意思是：

- 当 Python 把 `chat_engine_factory` 传进来时
- Rust 不只是把“函数对象”保存下来
- 还把这个 async callback 运行时所需要的 Python/async 上下文一并保存

这是后面 Rust 再回调 Python async 函数时必须要用到的。

如果没有这一步，Rust 后面很难正确地把 Python coroutine 转成 Rust future。

所以第一层桥接，其实在 `EntrypointArgs::new(...)` 就已经开始了。

#### 三、`make_engine(...)` 本体先干什么

真正的 `make_engine` 函数在：

- `lib/bindings/python/rust/llm/entrypoint.rs`

它一开始做的事情，不是立刻构建 engine，而是先组一个：

- `LocalModelBuilder`

然后把参数一项项灌进去：

- `model_name(...)`
- `endpoint_id(...)`
- `context_length(...)`
- `request_template(...)`
- `kv_cache_block_size(...)`
- `router_config(...)`
- `migration_limit(...)`
- `http_host(...)`
- `http_port(...)`
- `http_metrics_port(...)`
- `tls_cert_path(...)`
- `tls_key_path(...)`
- `extra_engine_args(...)`
- `runtime_config(...)`
- `namespace(...)`
- `namespace_prefix(...)`

这说明 `make_engine` 的第一职责其实是：

- 把 Python 的入参转成 Rust 内部的 model/engine builder 配置

你可以把它想成：

- “把 Python 世界的配置，翻译成 Rust 世界里能消费的建模对象”

#### 四、为什么它还会处理模型下载

接着你会看到这段逻辑：

- 如果 `model_path` 已经在本地存在，就直接用
- 否则调用 `LocalModel::fetch(...)`

这个设计说明：

- `make_engine` 不只是“创建 engine”
- 它还负责“确保本地模型资源已经就绪”

也就是说，它是在做一件更大的事：

- build local model
- then select engine

不是纯粹 new 一个对象那么简单。

所以如果你要给 `make_engine` 起一个更准确的中文名字，它其实更像：

- “构建并准备好 frontend backend engine”

#### 五、`make_engine` 的真正分叉在 `select_engine(...)`

`make_engine(...)` 前半段组完 `local_model` 之后，会进入：

- `select_engine(distributed_runtime, args, local_model)`

这里才是真正决定：

- 到底要生成哪种 engine 配置

它目前主要按 `EngineType` 分成：

1. `Echo`
2. `Dynamic`
3. `Mocker`

而 frontend 这条线最重要的是：

- `EngineType::Dynamic`

因为你在 `frontend/main.py` 里传的是：

```python
EntrypointArgs(EngineType.Dynamic, ...)
```

#### 六、`Dynamic` 分支到底做了什么

在 `select_engine(...)` 里，`Dynamic` 分支最核心的代码是：

```rust
let chat_engine_factory = args.chat_engine_factory.map(py_engine_factory_to_callback);
RsEngineConfig::Dynamic {
    model: Box::new(local_model),
    chat_engine_factory,
}
```

这一段可以直接翻译成人话：

1. 如果 Python 传进来了 `chat_engine_factory`
2. 就把它包装成 Rust 侧的 callback
3. 再塞进 `RsEngineConfig::Dynamic`

所以这里的关键不是“Dynamic engine 本身做了什么”，而是：

- Python 的工厂函数在这里被 Rust 接管了

#### 七、`py_engine_factory_to_callback(...)` 是整个桥接的核心

如果说你现在最该认真读哪一段 Rust，这一段就是第一优先级。

它的职责是：

- 把 Python async callback
- 变成 Rust 可以 `await` 的 callback

整个过程可以拆成 4 步。

##### 第 1 步：Rust 拿到模型实例信息

callback 的入参是：

- `RsModelCardInstanceId`
- `RsModelDeploymentCard`

也就是说，当 Rust runtime 发现某个模型实例可用时，它会拿着这两个对象去触发 callback。

##### 第 2 步：Rust 重新拿 GIL，把 Rust 对象包回 Python 对象

在 `Python::with_gil(...)` 里，它会把：

- `instance_id`
- `card`

重新包装成 Python 可见对象：

- `ModelCardInstanceId`
- `ModelDeploymentCard`

这样 Python 的 `chat_engine_factory(instance_id, mdc)` 才能被正常调用。

##### 第 3 步：Rust 调用 Python async 函数，拿到 coroutine

这里不是直接拿结果，而是：

- 调 Python callback
- 得到 coroutine

然后用：

- `pyo3_async_runtimes::into_future_with_locals(...)`

把这个 Python coroutine 转成 Rust future。

这一步非常关键，因为：

- Python async 函数返回的是 coroutine
- Rust runtime 要消费的是 future

所以这里其实是在做：

- Python async -> Rust awaitable

##### 第 4 步：Rust await Python 结果，再把结果提取成 `PythonAsyncEngine`

当 Python coroutine 执行完后，Rust 会拿到一个 Python 对象结果。

然后它会尝试：

- `extract` 出 `PythonAsyncEngine`

最后把这个 `PythonAsyncEngine` 包进：

- `Arc`

变成 Rust 侧能长期持有和驱动的 engine。

所以整条链其实是：

```text
Rust 发现模型实例
  -> 调 Python chat_engine_factory(instance_id, mdc)
  -> Python 创建 SglangProcessor / router / tokenizer
  -> Python 返回 PythonAsyncEngine
  -> Rust 拿到这个 engine 并接到 Dynamic engine 体系里
```

#### 八、这和你前面看的 `SglangEngineFactory.chat_engine_factory(...)` 是怎么对上的

这时候你就能把 Rust 和 Python 彻底对上了。

Python 那边你已经看过：

- `SglangEngineFactory.chat_engine_factory(...)`

它干的事情是：

1. 加载 tokenizer
2. 根据 `instance_id.triple()` 解析：
    - `namespace`
    - `component`
    - `endpoint`
3. 构造：
    - `generate_endpoint = runtime.endpoint(...)`
4. 根据 router mode 创建：
    - `KvRouter(...)`
    - 或 `generate_endpoint.client(...)`
5. 创建：
    - `SglangProcessor(...)`
6. 最后返回：
    - `PythonAsyncEngine(gen.generator, loop)`

而 Rust 这边做的就是：

- 调这个 Python factory
- 把它返回的 `PythonAsyncEngine` 接到 Rust Dynamic engine 配置里

所以这两边是严丝合缝接起来的。

#### 九、为什么这里非得设计成“Rust 回调 Python”

因为这个项目的职责分层就是这样：

- Rust 负责 runtime / engine / HTTP / discovery 的底层框架
- Python 负责具体后端适配和请求处理逻辑

如果 frontend 只支持一种后端，全部写死在 Rust 里当然也行。  
但 Dynamo 要支持：

- SGLang
- vLLM
- 以及其他 Python 侧适配

那更自然的做法就是：

- Rust 提供通用框架
- Python 提供具体 chat engine 工厂

这样 Rust runtime 不需要知道：

- 怎么初始化 SGLang tokenizer
- 怎么构造 vLLM processor
- 这些特定后端的细节

它只需要知道：

- “我有一个 Python callback，它能给我一个 `PythonAsyncEngine`”

#### 十、所以 `make_engine` 最终产出的到底是什么

Python 视角里，它返回的是：

- `EngineConfig`

但这个 `EngineConfig` 里面实际包着的是：

- `RsEngineConfig`

也就是说，`make_engine(...)` 的真正输出不是“已经开始处理请求的 engine”，而是：

- 一份 Rust 侧可运行的 engine 配置对象

后面你在 `frontend/main.py` 里看到：

```python
await run_input(runtime, "http", engine)
```

这个 `engine`，其实就是前面 `make_engine(...)` 构建出来的 `EngineConfig`。

所以：

- `make_engine` 负责“把 engine 装起来”
- `run_input` 负责“让这个 engine 真正开始接 HTTP 请求”

#### 十一、你可以把它拆成两个阶段来记

##### 阶段 1：构建阶段

由 `make_engine(...)` 完成：

- 处理模型路径
- 下载模型
- 构建 local model
- 选择 Dynamic/Echo/Mocker engine
- 把 Python `chat_engine_factory` 转成 Rust callback

##### 阶段 2：运行阶段

由 `run_input(...)` 驱动：

- 启动 HTTP / gRPC / text input
- 当请求到来时，借助前面配置好的 engine 去处理请求

这样你以后看 `frontend/main.py` 时就不会把：

- `make_engine`
- `run_input`

混成一件事。

#### 十二、最值得你现在记住的 3 个关键点

##### 1. `make_engine` 的核心不是“执行请求”，而是“构建 Rust engine 配置”

它做的是 build，不是 serve。

##### 2. `chat_engine_factory` 不是直接在 Python 里被 frontend 调用，而是被 Rust runtime 回调

这一点非常重要。

也就是说：

- frontend Python 只是把 factory 交给 Rust
- 真正调用这个 factory 的一方，是 Rust

##### 3. Python 返回的不是普通对象，而是 `PythonAsyncEngine`

Rust 只认这个统一抽象。

所以 Python 后端适配层最终都要归一到：

- `PythonAsyncEngine`

这就是跨语言边界的协议面。

#### 十三、最后给你一张最简图

```text
frontend/main.py
  -> EntrypointArgs(chat_engine_factory=Python callback)
  -> make_engine(runtime, args)

Rust make_engine:
  -> LocalModelBuilder
  -> build local model
  -> select_engine()
  -> Dynamic 分支
  -> py_engine_factory_to_callback(...)
  -> EngineConfig

运行时:
  -> Rust 发现模型实例
  -> 回调 Python chat_engine_factory(instance_id, mdc)
  -> Python 创建 SglangProcessor
  -> Python 返回 PythonAsyncEngine
  -> Rust 驱动这个 engine 处理请求
```

#### 最短总结

`make_engine` 在 Rust 里做的事情，本质上是：

- 把 Python 传进来的 frontend 配置，构造成 Rust 侧可运行的 Dynamic engine 配置

而 `chat_engine_factory` 的接入方式是：

- 先在 `EntrypointArgs` 里保存 Python callback 和 TaskLocals
- 再在 `select_engine()` 的 `Dynamic` 分支里转成 Rust callback
- 等模型实例被发现时，由 Rust 回调 Python
- Python 返回 `PythonAsyncEngine`
- Rust 再把它纳入自己的请求处理框架

### 问题 2

问题：

继续。

详细回答：

我们现在把上一段再往前推进一步，把：

- `make_engine(...)`

接到：

- `run_input(runtime, "http", engine)`

这一段。

如果先说结论：

**`run_input(...)` 本身不是 HTTP 服务器实现，它只是一个“输入模式分发器”；真正把 HTTP 请求接进来的，是 `lib/llm/src/entrypoint/input/http.rs` 里的 `http::run(...)`。**

所以这条链你可以先记成：

```text
frontend/main.py
  -> make_engine(...)
  -> run_input(runtime, "http", engine)

Rust:
  -> run_input(...)
  -> Input::Http
  -> http::run(...)
  -> HttpService::builder().build()
  -> run_watcher(...)
  -> 发现模型实例
  -> 注册/启用 HTTP endpoints
  -> HTTP 请求进入 OpenAI 路由
```

#### 一、`run_input(...)` 自己其实很薄

在：

- `lib/llm/src/entrypoint/input.rs`

里，`run_input(...)` 的逻辑非常直接：

```rust
match in_opt {
    Input::Http => {
        http::run(drt, engine_config).await?;
    }
    Input::Grpc => { ... }
    Input::Text => { ... }
    ...
}
```

所以它本质上只是：

- 根据输入模式做 dispatch

如果你从 frontend/main.py 看它，会觉得：

- `run_input(runtime, "http", engine)` 很像“开始处理请求”

这句话并没有错，但更准确地说是：

- “进入 HTTP 模式，并把后续交给 `http::run(...)`”

#### 二、`Input::Http` 真正进入的是 `http::run(...)`

在：

- `lib/llm/src/entrypoint/input/http.rs`

里，`http::run(...)` 才是 HTTP 入口装配的真正核心。

这个函数要解决的是：

1. 怎么把 `EngineConfig` 挂到 HTTP service 上
2. 怎么让 HTTP service 知道当前有哪些模型实例可用
3. 怎么在模型实例动态增减时更新可用 endpoints

也就是说，这一层已经从：

- “构建 engine”

过渡到：

- “让 engine 真正开始对外提供 HTTP 服务”

#### 三、`http::run(...)` 先构建 `HttpService`

它一进来先从 `engine_config.local_model()` 里读：

- `http_port`
- `http_host`
- `tls_cert_path`
- `tls_key_path`
- request template

然后组一个：

- `HttpService::builder()`

也就是说，真正的 OpenAI 兼容 HTTP server 底层是：

- `HttpService`

不是 Python 那边自己手搓的 FastAPI/Flask。

所以你现在可以明确一件事：

- frontend 的 HTTP 服务核心实现，主要在 Rust

Python 负责的是：

- 处理具体模型请求逻辑

Rust 负责的是：

- HTTP 服务框架

#### 四、它还把 discovery、metrics、request template 都注入进 HTTP service

在 builder 上你会看到这些关键设置：

- `cancel_token(...)`
- `with_request_template(...)`
- `drt_metrics(...)`
- `drt_discovery(...)`

这说明这个 HTTP service 不是个单纯的“收 HTTP 请求然后调函数”的小服务，而是和整个 Dynamo runtime 深度绑定的：

- 它知道 DRT metrics
- 它知道 DRT discovery
- 它知道 request template

所以这一步本质上是在做：

- 把 runtime 世界的能力接到 HTTP 世界上

#### 五、最关键的分叉：`EngineConfig::Dynamic`

接下来 `http::run(...)` 会按 `engine_config` 分支。

对你现在的 frontend 主线来说，最关键的是：

- `EngineConfig::Dynamic`

在这个分支里，它会做几件非常重要的事。

##### 1. 创建 `HttpService`

先 build 出 HTTP service 本体。

##### 2. 读取 router/migration 配置

从 model 上拿：

- `router_config`
- `migration_limit`

这说明 HTTP service 后面不只是把请求丢给某个固定模型，而是要支持：

- 动态模型发现
- 动态路由
- 迁移能力

##### 3. 构造 namespace filter

这一步是为了后面的 model watcher：

- 只监听当前 frontend 关心的 namespace / namespace_prefix

##### 4. 调 `run_watcher(...)`

这是 Dynamic 模式里最关键的一步。

因为 Dynamic frontend 和 in-process frontend 最大的区别就在这里：

- 它不是一启动就已经知道所有模型和 engine
- 它需要通过 discovery 去观察有哪些模型实例注册进来

所以它会启动 watcher，持续看：

- 当前有哪些模型上线/下线

#### 六、`run_watcher(...)` 的作用是什么

如果一句话概括：

**`run_watcher(...)` 负责把 runtime discovery 里动态出现的模型实例，转成 HTTP service 可用的 model/endpoints。**

它里面最关键的对象有两个：

1. `ModelWatcher`
2. `ModelManager`

你可以先这样理解：

- `ModelWatcher` 负责监听 discovery 世界的变化
- `ModelManager` 负责保存 HTTP service 当前可用的模型/engine 映射

#### 七、这里和前面的 `chat_engine_factory` 是怎么接上的

`run_watcher(...)` 里会构造：

- `ModelWatcher::new(...)`

并把这些东西传进去：

- `runtime`
- `model_manager`
- `router_config`
- `migration_limit`
- `chat_engine_factory`

这说明：

- 前面在 `make_engine(...)` 里包装好的 `chat_engine_factory`
- 到这里真正进入了“动态模型发现 -> 动态创建 engine”的链路

所以你现在要建立一个很重要的理解：

**`chat_engine_factory` 不是在 HTTP 请求到来时临时直接调用的，它更像是在“模型被 watcher 发现/注册时”由 Rust 动态触发的。**

也就是说，模型实例和请求入口之间还有一层：

- model discovery / watcher / manager

#### 八、watcher 具体在监听什么

在 `run_watcher(...)` 里，Rust 会用 runtime 的 discovery client 做：

- `list_and_watch(AllModels, ...)`

也就是说它会：

1. 先列出当前已有模型
2. 再持续 watch 后续新增/删除

所以当一个 worker 注册模型实例时，这条 watcher 线最终就能感知到。

你可以把它理解成：

```text
worker 注册模型实例
  -> discovery backend 更新
  -> ModelWatcher 收到新增事件
  -> Rust 侧知道“有新模型实例了”
  -> 调 chat_engine_factory(...)
  -> 生成对应 PythonAsyncEngine
  -> 塞进 ModelManager
  -> HTTP service 现在可以把请求路由给这个模型
```

#### 九、`update_http_endpoints(...)` 是干什么的

在 watcher 旁边，它还起了另一个 task：

- 专门根据 model update 来启用/关闭 HTTP endpoint

比如：

- 某个模型支持 `chat/completions`
- 某个模型支持 `completions`

它就会动态：

- `enable_model_endpoint(endpoint_type, true/false)`

这说明 HTTP service 暴露哪些 OpenAI 风格 endpoint，不是完全静态的，而是会随着当前可用模型类型动态变化。

这点很有意思，因为它意味着：

- frontend HTTP 层不只是固定转发器
- 它会根据模型能力动态收敛可用 API 面

#### 十、那请求真正进入 `/v1/chat/completions` 后会去哪

到这一步，HTTP service 已经起来了，model watcher 也在后台运行了。

接下来外部请求真正进来时，就会进入：

- `lib/llm/src/http/service/openai.rs`

那里才是 OpenAI 风格 HTTP 路由与 handler 真正处理请求的地方。

也就是说：

- `frontend/main.py` 负责启动
- `make_engine` 负责构建 engine
- `run_input/http::run` 负责启动 HTTP service 和动态模型管理
- `http/service/openai.rs` 负责实际处理 `/v1/chat/completions`

所以你现在已经快接近“真正处理一次 HTTP 请求”的入口了。

#### 十一、把这条链按职责重新整理一下

##### 第 1 层：Python frontend 装配层

在 `frontend/main.py`：

- 生成 `EntrypointArgs`
- 调 `make_engine`
- 调 `run_input("http", ...)`

##### 第 2 层：Rust engine 构建层

在 `entrypoint.rs`：

- 构建 `EngineConfig::Dynamic`
- 包装 `chat_engine_factory`

##### 第 3 层：Rust HTTP 服务层

在 `entrypoint/input/http.rs`：

- 创建 `HttpService`
- 启动 `ModelWatcher`
- 动态管理可用模型和 endpoints

##### 第 4 层：Rust OpenAI HTTP 路由层

在 `http/service/openai.rs`：

- 真正处理 `/v1/chat/completions`

##### 第 5 层：Python 后端处理层

最终再回到：

- `SglangEngineFactory.chat_engine_factory(...)`
- `SglangProcessor`

#### 十二、你现在最该怎么继续追

如果你要继续沿着“一个请求怎么进 frontend”这条线往下走，我建议下一站不要再回 Python，而是先看：

- `lib/llm/src/http/service/openai.rs`

原因是：

- 我们现在已经把请求入口推进到 HTTP service 了
- 下一步最自然的问题就是：
    - `/v1/chat/completions` 到底是在哪个 Rust handler 里被接住的？

等你把这个入口看完，再回头接：

- `SglangProcessor.generator`

那时候你会更清楚：

- Rust HTTP 层和 Python 生成层到底在哪一层交接

#### 最短总结

`run_input(runtime, "http", engine)` 之后，真正发生的是：

- Rust `run_input(...)` 把输入模式分发到 `http::run(...)`
- `http::run(...)` 创建 `HttpService`
- 对 `Dynamic` engine 启动 `ModelWatcher`
- 通过 discovery 动态发现模型实例
- 通过 `chat_engine_factory` 动态生成 Python engine
- 再把外部 `/v1/chat/completions` 请求接进 Rust 的 OpenAI HTTP service

所以现在这条链已经推进到：

- **HTTP server 已经启动，下一步该看 OpenAI 路由入口了**

### 问题 3

问题：

继续。

详细回答：

我们现在继续往前走，真正进入：

- `/v1/chat/completions`

这一层。

如果先说一句最关键的话：

**外部 HTTP 请求真正打到 frontend 后，第一站是 Rust 的 OpenAI HTTP 路由；它会先做模板填充、校验、metrics/context 准备，然后从 `ModelManager` 里取到对应 engine，最后统一调用 `engine.generate(request)`。**

所以你现在可以把链路继续更新成：

```text
HTTP client
  -> /v1/chat/completions
  -> Rust handler_chat_completions(...)
  -> Rust chat_completions(...)
  -> state.manager().get_chat_completions_engine_with_parsing(model)
  -> engine.generate(request)
  -> 后面才进入具体 Dynamic/Python engine
```

这一步非常关键，因为它告诉你：

- HTTP 请求并不是直接先进入 Python
- 而是先进入 Rust 的 OpenAI service 层

#### 一、真正的 HTTP 入口函数是谁

在：

- `lib/llm/src/http/service/openai.rs`

你会看到两个很关键的函数：

1. `handler_chat_completions(...)`
2. `chat_completions(...)`

它们的关系可以理解成：

- `handler_chat_completions(...)`：HTTP 框架层入口
- `chat_completions(...)`：真正的业务处理主函数

也就是说：

- 外部请求先被 `handler_chat_completions` 接住
- 然后再转进 `chat_completions`

#### 二、`handler_chat_completions(...)` 做了什么

这个函数做的事很像典型的 HTTP handler 外壳：

##### 1. 检查 service ready

```rust
check_ready(&state)?;
```

所以如果 frontend 还没 ready，会在这里直接拦住。

##### 2. 处理 header 路由覆盖

```rust
request.nvext = apply_header_routing_overrides(...)
```

说明：

- 请求头也可能影响 routing 行为

##### 3. 创建 request_id 和 context

它会生成：

- request_id
- context

并创建 connection monitor。

这说明在 HTTP 层一开始，就已经把：

- tracing
- cancellation
- metrics

这些基础设施接进来了。

##### 4. 起一个 task 跑真正的 `chat_completions(...)`

```rust
tokio::spawn(chat_completions(...))
```

这一步很重要，因为它说明：

- 真正长耗时生成任务是在单独 task 里跑的
- 外层 handler 负责连接生命周期和取消控制

所以 `handler_chat_completions(...)` 更像：

- 请求外壳
- 生命周期管理器

而真正的请求业务在：

- `chat_completions(...)`

#### 三、`chat_completions(...)` 才是最关键的请求处理主线

你可以把这个函数当成：

- frontend 请求在 Rust HTTP 层的真正主流程

它的步骤很有层次，特别适合你现在建立心智模型。

#### 四、第 1 步：先处理模板和默认值

一进来它会先做：

- 读取 `stream`
- 应用 request template

例如：

- 如果 request 里 model 为空，就从 template 补
- 如果 temperature/max_completion_tokens 没写，就用 template 默认值

这一点很重要，因为它说明：

- 在 frontend 里，请求刚到达时并不是最终形态
- Rust HTTP 层还会先做一轮规范化

所以当你以后看请求链路时，不要以为客户端发来的 JSON 就是最终发给 worker 的那份 request。

中间还有 frontend 层的预处理。

#### 五、第 2 步：确定最终 `model`

模板应用完以后，它会拿到：

- `model = request.inner.model.clone()`

这一点特别关键，因为后面几乎所有事情都围绕这个 model 名来做：

- metrics label
- queue guard
- engine lookup

所以从这里开始，frontend 就已经明确知道：

- 这次请求要打到哪个逻辑模型

#### 六、第 3 步：做一轮完整校验

在真正调用 engine 之前，Rust HTTP 层会做多轮 validation：

- unsupported fields
- required fields
- stream_options 检查
- generic field validation

这一层的含义是：

- frontend 先在 HTTP 协议层保证 OpenAI 请求语义是合理的
- 后面的 engine/worker 不需要再承担最外层的协议校验负担

所以这里也是一个清晰分层：

- Rust HTTP service：协议与 API 层校验
- Python/SGLang processor：业务生成层

#### 七、第 4 步：从 `ModelManager` 拿到真正的 engine

这一句是整个函数里最关键的交接点之一：

```rust
let (engine, parsing_options) = state
    .manager()
    .get_chat_completions_engine_with_parsing(&model)
```

你可以把它翻译成人话：

- “根据这次请求的 model 名，从当前 frontend 持有的模型注册表里，取出一个可以处理 chat completions 的 engine”

这里的 `state.manager()` 就是前面 `run_watcher(...)` 持续往里塞模型实例的那个：

- `ModelManager`

所以现在整条链终于闭环了：

```text
worker 注册模型实例
  -> watcher 发现
  -> chat_engine_factory 生成 PythonAsyncEngine
  -> 放进 ModelManager

HTTP 请求到来
  -> chat_completions(...)
  -> 从 ModelManager 取出 engine
```

所以 `ModelManager` 就是：

- 动态模型发现世界
- 和实际 HTTP 请求处理世界

之间的汇合点。

#### 八、第 5 步：统一调用 `engine.generate(request)`

一旦拿到 engine，最关键的一句就来了：

```rust
let stream = engine.generate(request).await.map_err(...)
```

这一句特别值得你记住。

因为它说明对于 Rust HTTP 层来说，后面的所有后端实现其实都被抽象成了同一种能力：

- 给我一个 request
- 返回一个 stream

也就是说：

- 不管底下是 SGLang 还是 vLLM
- 不管 engine 最终是 Rust 内置的还是 Python 动态生成的

到 Rust HTTP service 这一层，它只认：

- `engine.generate(...)`

所以这里就是协议层和执行层之间最关键的抽象边界。

#### 九、这个 `engine.generate(request)` 后面会去哪

这是你现在最关心的问题之一。

对 Dynamic frontend 来说，这个 `engine` 很大概率最终就是：

- 前面通过 `chat_engine_factory` 动态创建出来的 `PythonAsyncEngine`

所以：

- Rust HTTP service 调 `engine.generate(request)`

在底层就会进一步进入：

- PythonAsyncEngine
- 然后进入 `SglangProcessor.generator`

所以你现在可以把真正的跨语言交界点定在这里：

```text
Rust HTTP 层
  -> engine.generate(request)
  -> Dynamic engine / PythonAsyncEngine
  -> Python SglangProcessor
```

这就是后面我们要继续追的重点。

#### 十、streaming 和 non-streaming 在这里怎么分

这个函数还有一个很重要的点：

- 无论客户端是不是 streaming 请求
- 它都会先走统一的 streaming engine 逻辑

也就是说：

- `engine.generate(request)` 统一返回的是 stream

然后：

##### 如果 `streaming == true`

- Rust 把 stream 转成 SSE
- 并支持早期工具调用 dispatch、reasoning dispatch、disconnect monitor

##### 如果 `streaming == false`

- Rust 先检查 stream 的第一个 event 有没有 backend error
- 然后把整个 stream fold 成一个单次响应

这个设计很漂亮，因为它说明：

- 底层生成统一按流处理
- HTTP 层再决定对客户端呈现成 streaming 还是 unary

所以 frontend 的 OpenAI API 兼容层，其实做了很强的一层协议适配。

#### 十一、为什么这里能处理工具调用和 reasoning dispatch

你在 streaming 分支里会看到：

- `tool_call_dispatch`
- `reasoning_dispatch`

这些都是 SSE 的 side-channel event。

这说明 Rust HTTP 层不仅是在“原样转发 token”，它还在做：

- OpenAI 风格事件流包装
- tool call 的早期派发
- reasoning 内容的聚合和派发

也就是说：

- worker 返回的原始流
- 到客户端看到的 SSE 流

中间还有一层比较重的 frontend HTTP 语义加工。

#### 十二、这一层对你的学习意味着什么

到这里，你现在应该明确三件事。

##### 1. frontend 的真正 HTTP 入口在 Rust

不是在 Python 的 `main.py`，也不是在 `sglang_processor.py`。

##### 2. `ModelManager` 是动态发现和请求执行的汇合点

这非常关键。

##### 3. `engine.generate(request)` 是 Rust -> Python/后端执行层的统一交接点

所以你接下来再往下追时，应该优先盯住：

- 这个 `engine.generate(request)` 对 Dynamic/PythonAsyncEngine 来说是怎么实现的

#### 十三、你现在最适合的下一步

如果沿着“一个请求怎么进 frontend”继续往下走，下一站最自然就有两条路：

##### 路线 A：继续留在 Rust

看：

- Dynamic engine 对 `generate(request)` 的具体实现

##### 路线 B：顺着跨语言边界进入 Python

看：

- `PythonAsyncEngine`
- `SglangProcessor.generator`

对你现在的目标来说，我建议优先走路线 B，因为你已经很明确想知道：

- 请求最后怎么去对应 worker

而这条逻辑更接近：

- `SglangProcessor`
- `router.generate(...)`

#### 最短总结

`/v1/chat/completions` 的请求进入 frontend 后，Rust HTTP 层会先做：

1. 模板补全
2. 协议校验
3. metrics/context/cancellation 准备
4. 从 `ModelManager` 里取出对应 model 的 engine
5. 统一调用：
    - `engine.generate(request)`

所以这一层最关键的交界点是：

- **`state.manager().get_chat_completions_engine_with_parsing(model)`**
- **`engine.generate(request)`**

而你接下来真正要追的，就是：

- **这个 `generate(request)` 如何跨到 Python 的 `SglangProcessor`**

### 问题 4

问题：

继续。

详细回答：

我们现在正式跨过 Rust/Python 边界，去看：

- Rust 的 `engine.generate(request)`

到底是怎么落到：

- Python 的 `SglangProcessor.generator(...)`

上的。

如果先说最核心的结论：

**对 Dynamic frontend 来说，Rust 拿到的 engine 本质上是 `PythonAsyncEngine`；当 Rust 调 `engine.generate(request)` 时，它会把 Rust request 序列化成 Python 对象，调用 Python async generator，并把 Python 产出的每个 item 再反序列化回 Rust stream。**

这就是整个跨语言桥的核心。

你可以先把这条链记成：

```text
Rust chat_completions(...)
  -> engine.generate(request)
  -> PythonAsyncEngine.generate(...)
  -> PythonServerStreamingEngine.generate(...)
  -> pythonize(request)
  -> 调 Python generator(request, context=...)
  -> Python: SglangProcessor.generator(...)
  -> yield dict chunks
  -> Rust depythonize(item)
  -> 重新变回 Rust Annotated stream
```

#### 一、`PythonAsyncEngine` 在 Rust 里是什么

实现位置在：

- `lib/bindings/python/rust/engine.rs`

最外层结构是：

```rust
pub struct PythonAsyncEngine(PythonServerStreamingEngine);
```

它实现了 Rust 的：

- `AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>`

也就是说，对 Rust 上层来说，`PythonAsyncEngine` 看起来就是一个普通 Rust async engine。

所以 Rust HTTP 层不需要知道后面是不是 Python，它只需要知道：

- 我可以调用 `generate(request)`

这就是为什么前面在 `openai.rs` 里它能统一写成：

```rust
engine.generate(request).await
```

#### 二、真正干活的是 `PythonServerStreamingEngine.generate(...)`

`PythonAsyncEngine.generate(...)` 本身很薄，它只是转调：

- `self.0.generate(request).await`

真正的桥接逻辑都在：

- `PythonServerStreamingEngine.generate(...)`

这里可以分成 6 步来看。

#### 三、第 1 步：Rust 把 request/context 拆开

一进来它先做：

- 从 `SingleIn<Req>` 里拿出 request 和 context
- 生成 `request_id`
- 拿 tracing context

这说明跨语言边界前，Rust 先把：

- request 数据
- cancellation context
- tracing 上下文

都整理好了。

所以 Python 不是凭空拿到一个 dict，而是拿到一个带上下文的信息包。

#### 四、第 2 步：创建 Rust channel，准备接 Python stream

它会创建一个：

- `mpsc::channel::<Annotated<Resp>>(128)`

这个 channel 的意义是：

- Python async generator 会不断产出 item
- Rust 这边需要把这些 item 转回自己的 `Annotated<Resp>` stream

所以中间需要一条异步桥。

你可以把它理解成：

- Python producer
- Rust consumer

之间的缓冲通道。

#### 五、第 3 步：Rust 把 request `pythonize` 成 Python 对象，并调用 Python generator

这是最关键的一步之一。

在 `spawn_blocking + Python::with_gil(...)` 里，Rust 会做：

##### 1. `pythonize(py, &request)`

把 Rust request 变成 Python 对象。

对你当前这条链来说，最终在 Python 那边看到的，就会像：

- `dict[str, Any]`

所以这就是为什么 `SglangProcessor.generator(...)` 的签名是：

```python
async def generator(self, request: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]
```

因为 Rust 已经先把请求转成了 Python dict 风格对象。

##### 2. 创建 Python `Context`

Rust 还会构造：

- `Context::new(ctx_python.clone(), current_trace_context)`

然后作为 kwarg：

- `context=...`

传给 Python generator。

这说明：

- Python 侧不仅拿到 request
- 还能拿到 cancellation / trace 等上下文对象

##### 3. 真正调用 Python generator

如果 Python callable 支持 `context` 参数，就这样调：

```rust
generator.call(py, (py_request,), Some(&kwarg))
```

否则就兼容老接口：

```rust
generator.call1(py, (py_request,))
```

也就是说，Rust 在这里真正调用的是：

- 你前面从 `SglangEngineFactory` 返回的那个 `PythonAsyncEngine(gen.generator, loop)` 里的 `gen.generator`

所以最终落点就是：

- `SglangProcessor.generator(...)`

#### 六、第 4 步：把 Python async generator 转成 Rust stream

Rust 调完 Python generator，不是直接拿结果，而是得到一个：

- Python async generator / coroutine stream

然后用：

- `pyo3_async_runtimes::tokio::into_stream_with_locals_v1(...)`

把它转成 Rust 可以 `next().await` 的 stream。

这一步非常关键，因为这是：

- Python async stream
- -> Rust async stream

的真正桥接点。

所以现在你可以明确地说：

- `PythonAsyncEngine` 的核心价值，就是把 Python async generator 变成 Rust `ResponseStream`

#### 七、第 5 步：Rust 一边消费 Python item，一边 `depythonize`

后面 Rust 会起一个 task，循环：

- `stream.next().await`

每拿到一个 Python item，就调用：

- `process_item::<Resp>(item).await`

这里最关键的是：

- `depythonize::<Resp>(&item.into_bound(py))`

也就是说：

- Python yield 出来的 dict / object
- 会被反序列化回 Rust 期望的响应结构

所以整个跨语言边界是双向对称的：

##### 请求方向

- Rust request -> `pythonize` -> Python dict/object

##### 响应方向

- Python yield item -> `depythonize` -> Rust response struct

这就是为什么 Python 侧虽然写起来像普通 dict 流，但最后 Rust HTTP 层依然能把它当成强类型 stream 来继续处理。

#### 八、第 6 步：最终返回 Rust `ResponseStream`

Rust 最后会把 channel 包成：

- `ReceiverStream`

再构造：

- `ResponseStream::new(...)`

所以从 Rust 上层看，整个 Python generator 已经被完全包装成一个标准 Rust streaming engine 输出。

也就是说，到了 `openai.rs` 那一层，它根本感觉不到“这里经过了 Python”。

#### 九、现在切到 Python 侧：`SglangProcessor.generator(...)` 是第一站

实现位置在：

- `components/src/dynamo/frontend/sglang_processor.py`

你可以把这个函数看成：

- “请求进入 Python 应用层后的第一个真正处理入口”

它的主逻辑很简单：

1. 如果没有 preprocess pool，就走：
    - `_generator_inner(request)`
2. 如果有 preprocess pool，就走：
    - `_generator_inner_pool(request)`

也就是说：

- `generator(...)` 自己更像一个总入口和调度器

真正的请求处理主线在：

- `_generator_inner(...)`

#### 十、`_generator_inner(...)` 做的第一件事：预处理请求

这里最关键的调用是：

```python
pre = preprocess_chat_request(...)
```

这一步会做：

- tokenizer 相关处理
- tool_call parser / reasoning parser 相关处理
- 把 OpenAI 风格 chat request 转成内部预处理结果

然后拿到：

- `pre.prompt_token_ids`

接着又会构造：

- `dynamo_preproc = _build_dynamo_preproc(...)`

这一步非常重要，因为它说明：

- Python 侧不会直接把原始 OpenAI chat request 发给 worker
- 它会先变成一份 Dynamo 内部生成请求

所以到这里，请求已经从：

- “HTTP/OpenAI 请求”

变成了：

- “Dynamo 内部 generate 请求”

#### 十一、然后进入 `_generate_and_stream(...)`

这是 Python frontend 到 worker 的关键一跳。

在：

- `_generate_and_stream(...)`

里，最关键的逻辑是：

##### 情况 1：KV router

```python
dynamo_stream = await self.router.generate(
    token_ids=tokens,
    model=dynamo_preproc["model"],
    stop_conditions=dynamo_preproc["stop_conditions"],
    sampling_options=dynamo_preproc["sampling_options"],
    output_options=dynamo_preproc["output_options"],
)
```

##### 情况 2：非 KV router

```python
dynamo_stream = await self.router.generate(
    dynamo_preproc, annotated=False
)
```

这一步的人话版就是：

- frontend Python 层终于把请求发给后端 router/client 了

也就是说，从“一个请求怎么进 frontend”的角度讲，到这里你已经到达：

- frontend 往 worker 发起内部 generate 调用的那一刻

#### 十二、这说明前后端的职责切分是什么

看到这里，你可以把职责切分得非常清楚：

##### Rust HTTP 层负责

- OpenAI 协议接入
- request_id / context / metrics / cancellation
- 统一 engine 抽象
- SSE / non-streaming HTTP 包装

##### Python frontend 层负责

- chat request 预处理
- tokenization / parser
- 构造 Dynamo 内部 generate 请求
- 调 router.generate(...)
- 处理 worker 返回的流并做 post-process

所以：

- Rust 更像入口协议层
- Python 更像业务请求处理层

#### 十三、你现在已经走到哪一步了

到这里，整条链已经推进到：

```text
HTTP /v1/chat/completions
  -> Rust openai.rs
  -> engine.generate(request)
  -> PythonAsyncEngine
  -> SglangProcessor.generator
  -> preprocess_chat_request(...)
  -> _build_dynamo_preproc(...)
  -> router.generate(...)
```

也就是说：

- 请求已经成功从 Rust HTTP 层跨进 Python frontend 业务层
- 并且已经到达“准备发给 worker”的最后一步了

#### 十四、下一步最自然应该看什么

现在最自然的下一站，就是继续往后看：

- `router.generate(...)` 到底怎么找到 worker
- `generate_endpoint.client(...)` 最终怎么发到 worker

如果你要继续沿着“请求最后怎么去对应的 worker”这条主线走，下一步最建议看：

1. `components/src/dynamo/frontend/sglang_processor.py`
    - `_generate_and_stream(...)` 后半段
2. `components/src/dynamo/frontend/sglang_processor.py`
    - `SglangEngineFactory.chat_engine_factory(...)` 里 router/client 是怎么构造的
3. 必要时回到 runtime endpoint/client 抽象

#### 最短总结

`engine.generate(request)` 跨到 Python 的过程，本质上是：

- Rust 用 `pythonize` 把 request 变成 Python 对象
- 调 Python async generator
- Python 在 `SglangProcessor.generator(...)` 里做 preprocess
- Python 调 `router.generate(...)`
- Python 再把流式结果一项项 `yield`
- Rust 用 `depythonize` 把这些 item 重新变成强类型响应 stream

所以：

- **`PythonAsyncEngine` 是 Rust 和 Python 请求流之间的桥**
- **`SglangProcessor.generator(...)` 是请求进入 Python frontend 业务层的第一站**

### 问题 5

问题：

继续。

详细回答：

我们现在继续把最后这一跳接上：

- frontend 里的 `router.generate(...)`
- 到底怎么找到某个 worker
- 最后怎么打到 `dynamo.sglang` 的 `generate` endpoint

如果先说结论：

**frontend 并不是直接拿 Pod IP 去发请求，而是先构造 `runtime.endpoint(namespace.component.endpoint)`，再从这个 endpoint 拿到一个 runtime `Client`/`PushRouter`；这个 client 会通过 discovery 持续观察有哪些 worker 实例，然后按 router mode 选中某个 instance，把请求发给它。worker 侧则通过 `serve_endpoint(handler.generate, ...)` 把自己的 `generate` 暴露到同一个 endpoint 名字上。**

你可以把这条链先记成：

```text
Python frontend
  -> runtime.endpoint("namespace.component.generate")
  -> endpoint.client(router_mode=...)
  -> PushRouter<Client>
  -> discovery watch 当前有哪些 worker instances
  -> 选中某个 instance
  -> 发 RPC 到该 instance

Python worker
  -> runtime.endpoint("namespace.component.generate")
  -> serve_endpoint(handler.generate, ...)
  -> 把自己注册成这个 endpoint 的一个实例
```

所以前后端真正对上的关键，不是：

- IP 地址

而是：

- 同一个 `namespace.component.endpoint` 名字

#### 一、frontend 这边是怎么构造目标 endpoint 的

在：

- `components/src/dynamo/frontend/sglang_processor.py`

的 `chat_engine_factory(...)` 里，你已经看到这段：

```python
(namespace_name, component_name, endpoint_name) = instance_id.triple()
generate_endpoint = self.runtime.endpoint(
    f"{namespace_name}.{component_name}.{endpoint_name}"
)
```

这一步非常重要。

它说明 frontend 不会说：

- “我要找 10.0.0.8:12345”

而是先说：

- “我要找 `某个 namespace` 下 `某个 component` 的 `某个 endpoint`”

这就是 Dynamo runtime 的命名寻址方式。

所以一个请求最后去哪个 worker，第一步并不是选 IP，而是先解析成：

- 逻辑 endpoint 名字

#### 二、`runtime.endpoint(...)` 在 Python 绑定里做了什么

这个方法的实现不在 Python，而在：

- `lib/bindings/python/rust/lib.rs`

它会：

1. 接收一个字符串路径
2. 支持：
    - `namespace.component.endpoint`
    - 或 `dyn://namespace.component.endpoint`
3. 拆成三段：
    - namespace
    - component
    - endpoint
4. 再沿着 Rust runtime 链式拿到真正的 `Endpoint` 对象：

```rust
let namespace = self.inner.namespace(namespace_name.to_string())?;
let component = namespace.component(component_name.to_string())?;
let endpoint = component.endpoint(endpoint_name.to_string());
```

所以 `runtime.endpoint(...)` 本质上是：

- 把 Python 层的 endpoint path 变成 Rust runtime 世界的 `Endpoint` 对象

#### 三、`generate_endpoint.client(...)` 返回的到底是什么

在 Python 绑定里，`Endpoint.client(...)` 的实现也在：

- `lib/bindings/python/rust/lib.rs`

这里它会做：

1. 先拿：
    - `inner.client().await`
2. 再构造：
    - `PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(...)`

最后返回 Python 层的：

- `Client`

但这个 Python `Client` 背后真正包着的，其实是 Rust 的：

- `PushRouter`

也就是说，frontend 里你看到的：

```python
router = await generate_endpoint.client(router_mode=...)
```

本质上是在创建一个：

- 可路由、可负载均衡、可发现实例的 Rust egress router

所以这个 `router.generate(...)` 不是简单的“调用某个固定远端”，而是：

- 先选实例
- 再发请求

#### 四、这个 `Client` / `PushRouter` 是怎么知道当前有哪些 worker 的

真正核心在：

- `lib/runtime/src/component/client.rs`

`Client::new(endpoint)` 之后，会做一件非常关键的事：

- 为这个 endpoint 建一个动态 instance source

它会向 discovery 发起：

- `DiscoveryQuery::Endpoint { namespace, component, endpoint }`

然后：

- `list_and_watch(...)`

也就是说，它不是只查一次，而是：

1. 先 list 当前已有实例
2. 再持续 watch 后续增删变化

这意味着 frontend 的 client 对某个 endpoint 的认知是动态更新的。

你可以把它理解成：

```text
我要访问 namespace.component.generate
  -> 去 discovery 查询这个 endpoint 当前有哪些实例
  -> 有新增/删除时持续同步
  -> 本地维护一个 instance 列表
```

所以 frontend 能找到 worker，不是因为 operator 把地址硬编码进去了，而是因为：

- worker 注册了 endpoint
- frontend client 通过 discovery watch 到了它

#### 五、`PushRouter` 是怎么选某个 worker instance 的

当 frontend 调：

- `router.generate(request)`

底层会进入：

- `lib/runtime/src/pipeline/network/egress/push_router.rs`

这里 `PushRouter.generate(...)` 会根据 `router_mode` 分流：

- `Random`
- `RoundRobin`
- `PowerOfTwoChoices`
- `KV`
- `Direct`

对非 KV 情况，常见的就是：

- `round_robin(request)`
- `random(request)`

比如 round-robin 会：

1. 取当前 `instance_ids_avail()`
2. 按计数器选一个实例
3. 调 `generate_with_fault_detection(instance_id, request)`

所以这里你可以理解为：

- frontend 的“选 worker”有一层通用 runtime 路由器

不是 Python 自己手工维护一堆地址列表。

#### 六、那 KV router 呢

你前面在 `SglangProcessor._generate_and_stream(...)` 里也看到了：

- 如果 `self.is_kv_router`
- 会走 `KvRouter.generate(...)`

这说明：

- KV router 是专门的一条更高级路由路径
- 它不是简单的 round-robin/random

但即便如此，它底层仍然是围绕某个：

- `generate_endpoint`

来工作的。

所以你现在先不要把注意力过早放到 KV 算法本身，先抓住这层抽象：

- 不管是普通 router 还是 KV router
- 它们都围绕同一个逻辑 endpoint 在选 worker 实例

#### 七、worker 这边是怎么把自己挂到这个 endpoint 上的

这一步在：

- `components/src/dynamo/sglang/init_llm.py`

里非常清楚。

你会看到：

```python
generate_endpoint = runtime.endpoint(
    f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
)
```

然后：

```python
await asyncio.gather(
    generate_endpoint.serve_endpoint(
        handler.generate,
        graceful_shutdown=True,
        metrics_labels=metrics_labels,
        health_check_payload=health_check_payload,
    ),
    register_model_with_readiness_gate(...),
)
```

这说明 worker 做了两件事：

1. 把自己的 `handler.generate` 暴露成这个 runtime endpoint 的服务端
2. 把模型实例注册进 discovery

这样 frontend 才能：

- 通过 model watcher 发现它
- 再通过 endpoint client 找到它

所以 worker 不是只“开了个 server”，还必须：

- 正确注册 discovery

#### 八、`serve_endpoint(...)` 本质上做了什么

在 Python 绑定里，`Endpoint.serve_endpoint(...)` 会：

1. 把 Python 的 `handler.generate` 包成：
    - `PythonAsyncEngine`
2. 创建 ingress handler
3. 调 endpoint builder
4. `register_local_engine(engine)`
5. `start()`

这一步的人话版就是：

- “把 Python 的 generate handler 挂到 runtime 的 endpoint 服务端上，并启动对外服务/注册流程”

所以你现在已经能把前后端完全对上了：

##### frontend

- `endpoint.client(...)`
- 这是调用方

##### worker

- `endpoint.serve_endpoint(...)`
- 这是服务方

两边共用的是同一个：

- `namespace.component.endpoint`

#### 九、所以一个请求最后去对应 worker 的完整逻辑是什么

现在我们把整个“frontend 找 worker”过程完整说一遍：

1. frontend 的 `SglangEngineFactory.chat_engine_factory(...)`
    - 根据 `instance_id.triple()` 构造：
        - `runtime.endpoint(namespace.component.endpoint)`
2. frontend 通过：
    - `generate_endpoint.client(router_mode=...)`
      拿到一个 runtime `Client/PushRouter`
3. 这个 client 在底层通过 discovery：
    - `list_and_watch(Endpoint{namespace,component,endpoint})`
      持续维护该 endpoint 下可用 worker instances 列表
4. 当请求到来时，frontend 调：
    - `router.generate(...)`
5. `PushRouter` 按 router mode 选中某个 instance
6. runtime request plane 把请求发到该 instance
7. worker 侧早已通过：
    - `generate_endpoint.serve_endpoint(handler.generate, ...)`
      把自己的 Python handler 挂在这个 endpoint 上
8. 请求最终进入：
    - `DecodeWorkerHandler.generate(...)`
      或对应 handler

所以最终“去对应的 worker”，不是靠：

- HTTP 层自己找 Pod IP

而是靠：

- endpoint 命名
- discovery watch
- runtime client/router
- worker endpoint registration

共同完成的。

#### 十、你现在最应该记住的 4 个关键词

如果你想快速记住这段链路，就记这 4 个点：

1. `runtime.endpoint(namespace.component.endpoint)`
    - 前后端统一命名空间
2. `endpoint.client(...)`
    - frontend 侧拿调用方 router
3. `list_and_watch(Endpoint query)`
    - 通过 discovery 动态发现 worker 实例
4. `endpoint.serve_endpoint(handler.generate, ...)`
    - worker 侧注册并暴露 generate 服务

#### 十一、从你当前学习目标看，下一站应该去哪

你现在已经基本知道：

- frontend 是怎么找到 worker 的

所以下一步最自然应该继续往 worker 内部走，也就是：

- `DecodeWorkerHandler.generate(...)`

换句话说，下一步最值得继续看的文件是：

- `components/src/dynamo/sglang/request_handlers/llm/decode_handler.py`

因为现在你已经把：

- operator -> pod
- frontend -> runtime -> worker endpoint

都串上了。

后面就该看：

- worker 收到 generate 请求后，到底怎么调用 `engine.async_generate()`

#### 最短总结

`router.generate(...)` 最终找到 worker 的方式，不是直接拿 IP，而是：

- frontend 先构造 `runtime.endpoint(namespace.component.endpoint)`
- 再通过 `endpoint.client(...)` 拿到动态 discovery client/router
- client 通过 `list_and_watch` 持续维护该 endpoint 下的 worker 实例列表
- 请求到来时按 router mode 选中某个 instance
- worker 侧通过 `endpoint.serve_endpoint(handler.generate, ...)` 提供对应服务

所以真正把 frontend 和 worker 接起来的核心，不是 Pod 地址，而是：

- **runtime endpoint 命名 + discovery + client/router + serve_endpoint**

### 问题 6

问题：

继续。

详细回答：

我们现在正式进入 worker 内部，回答这两个最关键的问题：

1. 请求到了 `DecodeWorkerHandler.generate(...)` 之后，怎么进入 `engine.async_generate()`？
2. worker 返回给 frontend 的流，长什么样？

如果先说一句最关键的话：

**`DecodeWorkerHandler.generate(...)` 的核心职责，就是把 frontend 发来的 Dynamo 内部 request 翻译成 SGLang `engine.async_generate(...)` 所需要的参数，然后把 SGLang 返回的流再包装成 Dynamo/frontend 能消费的输出流。**

所以这层的链路可以先记成：

```text
frontend router.generate(...)
  -> worker endpoint handler.generate(...)
  -> DecodeWorkerHandler.generate(request, context)
  -> _build_sampling_params(request)
  -> _get_input_param(request)
  -> engine.async_generate(...)
  -> _process_token_stream(...) / _process_text_stream(...)
  -> yield Dynamo/frontend 可理解的流式结果
```

#### 一、先看这个 handler 在哪里接住请求

你前面已经看到 worker 启动时在：

- `components/src/dynamo/sglang/init_llm.py`

里做了：

```python
handler = DecodeWorkerHandler(...)
await generate_endpoint.serve_endpoint(
    handler.generate,
    ...
)
```

这意味着：

- runtime endpoint 收到 `generate` 请求后
- 最终落到的 Python 函数就是：
    - `DecodeWorkerHandler.generate(...)`

所以这就是：

- frontend 到 worker 之后的第一站

#### 二、`DecodeWorkerHandler.generate(...)` 一进来先做什么

这个函数一开始并不会立刻调 SGLang，而是先做一轮参数整理。

##### 1. 记录 request/context 信息

比如：

- `context.id()`
- `trace_id`

说明 worker 层也保留了 request context 和 trace 传递能力。

##### 2. 构造 sampling 参数

它先调用：

```python
sampling_params = self._build_sampling_params(request)
```

这一步会把 frontend 传来的 request 映射成 SGLang 更关心的采样参数，比如：

- `temperature`
- `top_p`
- `top_k`
- `max_new_tokens`
- `ignore_eos`

而且它区分两种格式：

##### 情况 A：`skip_tokenizer_init == true`

说明 worker 接收到的是：

- token-based request

这时它会从：

- `sampling_options`
- `stop_conditions`

里取参数。

##### 情况 B：普通 OpenAI 风格请求

这时它会直接从 request 顶层字段取：

- `temperature`
- `top_p`
- `top_k`
- `max_tokens`

所以这一步本质上是在做：

- Dynamo request -> SGLang sampling 参数

#### 三、另一个关键转换：`_get_input_param(request)`

接着它还会调用：

- `self._get_input_param(request)`

这个函数定义在基类链上，它的职责就是：

- 从 request 中抽取真正给 SGLang engine 的输入字段

你可以把它理解成：

- 请求内容层的参数翻译

所以进入 `engine.async_generate(...)` 前，worker 至少会做两类翻译：

1. 输入内容翻译
2. 采样参数翻译

#### 四、`generate(...)` 为什么要分 aggregated 和 disaggregated 两条路径

这是 `DecodeWorkerHandler` 最重要的结构之一。

它先判断：

- `self.serving_mode == DisaggregationMode.DECODE`

然后分成：

##### 路径 1：disaggregated decode

要求 request 里必须带：

- `bootstrap_info`

这里会把：

- `bootstrap_host`
- `bootstrap_port`
- `bootstrap_room`

这些信息传给 SGLang。

这是给：

- prefill/decode 分离模式

用的。

##### 路径 2：aggregated

也就是你当前 `agg.yaml` 这类更关心的路径。

这条路不需要 bootstrap 信息，直接走本机聚合生成。

所以对你当前主线来说，最应该重点看的是：

- aggregated 分支

#### 五、aggregated 模式下，`engine.async_generate(...)` 是怎么调用的

在 aggregated 分支里，最关键的就是这一段：

```python
agg = await self.engine.async_generate(
    **input_param,
    image_data=image_data,
    sampling_params=sampling_params,
    stream=True,
    return_routed_experts=return_routed_experts,
    external_trace_header=trace_header,
    rid=trace_id,
    data_parallel_rank=dp_rank,
    **self._priority_kwargs(priority),
)
```

你可以把它拆成几个语义块。

##### 1. `**input_param`

这是请求本体输入，比如：

- input_ids
- text
- messages

具体形式取决于前面的 request 格式和转换逻辑。

##### 2. `sampling_params=sampling_params`

这是我们刚刚说的采样参数翻译结果。

##### 3. `stream=True`

这一点非常关键。

说明 worker 到 frontend 这一层，底层统一按流式生成。

这和前面 Rust HTTP 层的设计是对应上的：

- 不管最终客户端是不是 non-streaming
- 底层生成基本都统一按 stream 来处理

##### 4. `external_trace_header` / `rid`

这些是为了：

- tracing
- request identity

说明 worker 层生成调用也保留了跨组件 trace 传递。

##### 5. `data_parallel_rank`

这是配合 routing/并行策略的额外信息。

##### 6. `priority`

如果 engine 支持 priority，它也会透传进来。

所以这一句的本质就是：

- “把 frontend 内部 request 翻译完后，最终交给 SGLang engine 执行”

这就是整个 worker 逻辑里最核心的一跳。

#### 六、为什么这里 `await self.engine.async_generate(...)` 返回的还是一个流

这里要注意一个容易迷糊的点：

- `await self.engine.async_generate(...)`

返回的不是最终完整文本，而是：

- 一个异步流对象

后面代码会继续：

- `_process_token_stream(...)`
- 或 `_process_text_stream(...)`

去消费它。

所以更准确地说：

- `async_generate()` 是“创建生成流”
- 不是“一次性生成完整结果”

#### 七、worker 为什么还要再 `_process_*_stream(...)`

因为 SGLang engine 原生返回的流格式，不一定就是 frontend 直接想要的格式。

所以 worker handler 要做第二层包装。

它根据：

- `skip_tokenizer_init`

决定走：

##### 1. `_process_token_stream(...)`

给 token-based 场景用。

输出大概是：

- `token_ids`
- `finish_reason`
- `completion_usage`
- 可能还有 `disaggregated_params`

##### 2. `_process_text_stream(...)`

给 OpenAI 风格文本流场景用。

输出大概是：

- `id`
- `created`
- `choices`
- `delta.content`
- `finish_reason`
- `model`
- `object = "chat.completion.chunk"`

所以你可以理解成：

- worker 不是简单 pass through SGLang 原始结果
- 它在 worker 侧已经开始做 Dynamo/OpenAI 风格适配

#### 八、`_process_token_stream(...)` 做了什么

这条路径你虽然当前不一定最常用，但理解它很重要，因为它更接近内部协议。

它会：

1. 从第一个 response 的 `meta_info` 里提取 SGLang request ID
2. 监控 cancellation
3. 读取：
    - `output_ids`
    - `finish_reason`
    - `prompt_tokens`
    - `completion_tokens`
    - `cached_tokens`
4. 最终组织成一个内部输出 dict

最重要的几点是：

- `output_ids` 被直接作为新 token 段输出
- finish 时会附带 usage 统计
- 如果有 `routed_experts`，也会编码进结果里

所以这条路更像：

- Dynamo 内部 token stream 协议

#### 九、`_process_text_stream(...)` 做了什么

这条路径和你当前从 frontend 读进去的 OpenAI 请求更贴近。

它会：

1. 从 SGLang 返回里拿：
    - `text`
    - `index`
    - `meta_info.finish_reason`
2. 通过 `count` 和 `delta = text[count:]`
   计算出本次新增文本
3. 组装成：

```python
{
    "id": ...,
    "created": ...,
    "choices": [{
        "index": index,
        "delta": {"role": "assistant", "content": delta},
        "finish_reason": finish_reason_type,
    }],
    "model": ...,
    "object": "chat.completion.chunk",
}
```

这说明 worker 返回给 frontend 的已经非常接近 OpenAI chunk 了。

所以整个链路里协议适配其实分了两层：

##### Rust HTTP 层

- 负责最外层 OpenAI API 和 SSE

##### Python worker handler

- 已经把 SGLang 原始输出整理成接近 OpenAI chunk 的中间格式

这也是为什么 frontend 后面能比较顺利地继续包装和透传。

#### 十、context 在 worker 这里有什么用

你会看到这两个 `_process_*_stream(...)` 里都在用：

- `context.is_stopped()`
- `_cancellation_monitor(...)`

说明 worker 这层不是被动跑到底，而是支持：

- 客户端断开
- 上游取消
- graceful stop

这和前面 Rust HTTP 层的 disconnect monitor 是配套的。

所以取消链是贯通的：

- HTTP client 断开
- Rust connection monitor 感知
- Context stop
- Python worker stream 检查到 `is_stopped()`
- 中断生成/清理

#### 十一、把 worker 内部主线压缩成一句话

对 `agg.yaml` 这种 aggregated SGLang 路径来说，worker 侧主线就是：

```text
runtime endpoint 收到 generate 请求
  -> DecodeWorkerHandler.generate(...)
  -> 把 request 翻译成 input_param + sampling_params
  -> self.engine.async_generate(..., stream=True)
  -> _process_text_stream(...) / _process_token_stream(...)
  -> yield frontend 能消费的流式 chunk
```

#### 十二、这和你当前整条主线怎么接起来

现在你已经可以把从 frontend 到 worker 的主线串成：

```text
HTTP /v1/chat/completions
  -> Rust openai.rs
  -> engine.generate(request)
  -> PythonAsyncEngine
  -> SglangProcessor.generator
  -> preprocess_chat_request(...)
  -> router.generate(...)
  -> runtime endpoint client
  -> worker endpoint serve_endpoint(handler.generate)
  -> DecodeWorkerHandler.generate(...)
  -> engine.async_generate(...)
  -> _process_text_stream(...)
  -> 流回 frontend
```

到这里，你其实已经把“请求最后怎么走到 worker 并执行生成”这条主链打通了。

#### 十三、你接下来最值得继续看的是什么

如果你还想继续沿这条主链走，我建议下一步回到 frontend，重点看：

- `components/src/dynamo/frontend/sglang_processor.py`

里 `_generate_and_stream(...)` 的后半段。

因为现在 worker 这边你已经知道：

- 它吐出来的是什么 chunk

下一步最自然的问题就是：

- frontend 是怎么把这些 chunk 再处理成最终返回给客户端的 SSE / OpenAI 流的？

#### 最短总结

`DecodeWorkerHandler.generate(...)` 的核心就是：

- 把 frontend 发来的 Dynamo request 翻译成 SGLang `engine.async_generate(...)` 的参数
- 再把 SGLang 返回的原始流包装成 Dynamo/frontend 可继续消费的流

其中真正的执行点是：

- **`self.engine.async_generate(..., stream=True)`**

而真正决定返回格式的是：

- **`_process_token_stream(...)`**
- **`_process_text_stream(...)`**

### 问题 7

问题：

继续。

详细回答：

很好，我们现在把整条链最后一段补上：

- worker 已经把流吐回来了
- frontend 在 `sglang_processor.py` 里怎么把它加工成最终返回给客户端的输出

如果先说结论：

**在这条链里，worker 返回给 frontend 的通常是更偏内部协议的 token/usage/finish_reason 流；frontend 在 `_generate_and_stream(...)` 里做 token 聚合、detokenization/post-process、finish_reason 映射和最终 chunk 包装，产出的才是给 Rust HTTP 层继续向客户端输出的 OpenAI 风格流。**

所以链路的最后一段，你可以记成：

```text
worker -> token_ids / usage / finish_reason
  -> frontend SglangProcessor._generate_and_stream(...)
  -> SglangStreamingPostProcessor.process_output(...)
  -> OpenAI-style chunk dict
  -> Rust depythonize
  -> Rust SSE / non-stream response
  -> client
```

#### 一、先纠正一个很重要的认知

很多人看到 worker 里的 `_process_text_stream(...)`，会以为：

- worker 已经直接返回最终给客户端的 OpenAI chunk 了

但如果你对照 frontend 里的 `SglangProcessor._generate_and_stream(...)`，你会发现前端最稳定依赖的其实是：

- `engine_response["token_ids"]`
- `engine_response["finish_reason"]`
- `engine_response["completion_usage"]`

也就是说，从 frontend 的这条主线看，它更像是期待：

- token 级内部响应

然后再由 frontend 自己做最终 post-process。

所以你可以把这段理解成：

- worker 偏“内部生成输出”
- frontend 偏“最终 OpenAI 输出整形”

#### 二、frontend 收到 worker 返回后，第一步做什么

实现位置还是：

- `components/src/dynamo/frontend/sglang_processor.py`

在 `_generate_and_stream(...)` 里，拿到 `dynamo_stream` 之后，会先把每个 item 规范成：

- `engine_response`

逻辑是：

```python
if self.is_kv_router:
    engine_response = dynamo_response
elif hasattr(dynamo_response, "data"):
    engine_response = dynamo_response.data()
else:
    engine_response = dynamo_response
```

这一步的意义是：

- 屏蔽不同 router/transport 返回对象的差异
- 统一成 frontend 后续能消费的内部响应格式

所以从这一刻开始，frontend 不再关心这个响应最初是：

- KvRouter 直接返回的
- 还是 runtime annotated wrapper 包着的

它统一只看：

- `engine_response`

#### 三、frontend 最关心 worker 响应里的哪几个字段

后面这段最重要：

```python
new_ids = engine_response["token_ids"]
raw_finish = engine_response.get("finish_reason")
finish_reason = _map_finish_reason(raw_finish)

if usage := engine_response.get("completion_usage"):
    pending_usage = usage
```

这说明 frontend 后处理最关心的是三个东西：

1. `token_ids`
    - 新生成的 token 段
2. `finish_reason`
    - 是否结束、为什么结束
3. `completion_usage`
    - prompt/completion token 统计

所以这一步很像：

- worker 负责提供生成结果原料
- frontend 负责把这些原料拼成最终协议输出

#### 四、为什么 frontend 还要自己做 `_map_finish_reason(...)`

这里有一个非常细但很重要的点。

frontend 会用：

- `_map_finish_reason(raw_finish)`

去做一层映射。

例如：

- `"eos" -> "stop"`
- `"abort" -> "stop"`
- `"cancelled" -> "stop"`
- `"error:xxx" -> "error"`

这说明：

- worker/内部 runtime 的 finish_reason 语义
- 和 OpenAI API 对外暴露的 finish_reason 语义

不是完全同一套。

所以 frontend 在这里承担了：

- 内部状态语义
- 到 OpenAI 语义

之间的映射工作。

#### 五、为什么要 `pending_token_ids` 聚合，不是每来一个 token 就立刻吐

这段代码特别值得你记一下：

- `pending_token_ids`
- `stream_interval`
- `first_chunk`

逻辑是：

1. 第一块尽量快 flush（阈值 = 1）
    - 降低 TTFT
2. 后续按 `stream_interval` 聚合多个 token 再 flush
    - 减少 detokenization 和后处理开销

也就是说，frontend 在这里做了一个很工程化的 tradeoff：

- 第一 token 快速返回，优化交互体验
- 后续适当批处理，优化吞吐和 CPU 成本

所以这不是简单的“转发流”，而是带有性能策略的流处理。

#### 六、真正的核心后处理：`SglangStreamingPostProcessor.process_output(...)`

在达到 flush 条件后，frontend 会构造：

```python
mapped_response = {
    "token_ids": pending_token_ids,
    "finish_reason": finish_reason,
}
```

然后调用：

```python
choice = post.process_output(mapped_response)
```

这里的 `post` 就是：

- `SglangStreamingPostProcessor`

它是在前面 `preprocess_chat_request(...)` 之后，根据 tokenizer / tool parser / reasoning parser 创建出来的。

所以你可以理解成：

- preprocess 阶段负责把 OpenAI 请求转成内部生成请求
- postprocess 阶段负责把内部 token 输出转回 OpenAI 增量输出

这两个阶段正好一前一后配对。

#### 七、`choice` 其实已经很接近 OpenAI chunk 的核心部分了

当 `post.process_output(...)` 成功后，frontend 会得到：

- `choice`

然后再包成：

```python
dynamo_out = {
    "id": request_id,
    "choices": [choice],
    "created": created_ts,
    "model": request["model"],
    "object": "chat.completion.chunk",
}
if pending_usage:
    dynamo_out["usage"] = pending_usage
```

所以这一层其实已经非常接近最终客户端看到的流块了。

你可以把它理解成：

- `choice`：chunk 的核心业务内容
- `dynamo_out`：完整协议层包装

也就是说：

- Python frontend 在这里已经产出了 OpenAI 风格 chunk dict

#### 八、所以 frontend 这一步到底补了哪些协议字段

从 `dynamo_out` 来看，它补的是：

- `id`
- `choices`
- `created`
- `model`
- `object = "chat.completion.chunk"`
- 可选 `usage`

这说明 frontend 不只是“处理 token”，它还承担了最终面向客户端协议字段的拼装。

也就是说，在这条链里：

- worker 更偏生成执行者
- frontend 更偏 API 语义装配者

#### 九、`usage` 为什么是 frontend 在这里挂上去的

你会注意到：

- worker finish 时会带 `completion_usage`
- frontend 收到后先暂存在 `pending_usage`
- 真正 flush chunk 时，再把它挂到 `dynamo_out["usage"]`

这说明 usage 最后真正进入客户端响应，是在 frontend 这层完成的。

所以 usage 的归属链是：

```text
worker 统计
  -> 发给 frontend
  -> frontend 再注入最终输出 chunk
```

#### 十、Python frontend 产出的结果是怎么再回到 Rust HTTP 层的

这一点和前面 `PythonAsyncEngine` 的桥接对应起来就很清楚了。

frontend 这里 `yield dynamo_out` 之后：

- Rust `PythonServerStreamingEngine` 会拿到这个 Python dict
- 用 `depythonize::<Resp>(...)` 反序列化成 Rust 响应结构
- 再把它放回 Rust `ResponseStream`

所以从 Rust HTTP service 的视角看，它拿到的已经是：

- 一个标准化的 OpenAI chat completion streaming response 流

然后它才能继续做：

- SSE event 包装
- non-stream fold
- backend error 检查
- disconnect 处理

所以你现在可以明确：

- Python frontend 负责“业务与协议中间层”
- Rust HTTP 层负责“最终对外 HTTP/SSE 层”

#### 十一、把整条请求链闭环说一遍

到这里，这条主线已经可以完整闭环了：

```text
Client /v1/chat/completions
  -> Rust openai.rs handler
  -> ModelManager 取 engine
  -> engine.generate(request)
  -> PythonAsyncEngine
  -> SglangProcessor.generator
  -> preprocess_chat_request(...)
  -> _build_dynamo_preproc(...)
  -> router.generate(...)
  -> runtime client/discovery/router
  -> worker endpoint handler.generate
  -> DecodeWorkerHandler.generate(...)
  -> engine.async_generate(..., stream=True)
  -> worker _process_*_stream(...)
  -> frontend _generate_and_stream(...)
  -> SglangStreamingPostProcessor.process_output(...)
  -> yield OpenAI-style chunk dict
  -> Rust depythonize
  -> Rust SSE / unary response
  -> Client
```

如果你现在回头再看整个 frontend/worker 体系，应该会发现：

- 请求链已经真正从头到尾打通了

#### 十二、这一层最值得你记住的 5 个关键词

如果你想把 frontend 收尾这段牢牢记住，就记这 5 个点：

1. `pending_token_ids`
    - frontend 会聚合 token，再分批 flush
2. `_map_finish_reason(...)`
    - 内部 finish reason -> OpenAI finish reason
3. `SglangStreamingPostProcessor.process_output(...)`
    - token 输出 -> choice/delta
4. `dynamo_out`
    - 最终 Python frontend 产出的 chunk dict
5. `usage`
    - worker 提供，frontend 注入最终协议输出

#### 十三、你现在最适合怎么收尾这一阶段学习

从学习计划角度说，你现在已经完成了第二阶段里最重要的一件事：

- 把一次请求从 frontend 入口一路追到 worker，再追回 frontend 输出

所以接下来有两条非常自然的路线：

##### 路线 A：收束总结

你可以先把这条链写成你自己的时序图/脑图，彻底固化。

##### 路线 B：纵深专题化

从这条主线里挑一个专题深入，比如：

- KV router 选 worker 的算法
- discovery backend（K8s vs etcd）
- SglangStreamingPostProcessor 的 token/tool/reasoning 处理
- disaggregated prefill/decode 链路

如果按收益排序，我建议下一步优先深入：

- `KV router`

因为你现在已经知道请求是“怎么走”的了，接下来最值得问的就是：

- “为什么它最后选中了这个 worker？”

#### 最短总结

frontend 在 `SglangProcessor._generate_and_stream(...)` 里做的最后一层工作是：

- 接收 worker 的内部 token 流
- 聚合 token
- 映射 finish_reason
- 调 `SglangStreamingPostProcessor.process_output(...)`
- 包装成最终 OpenAI 风格 chunk dict

所以：

- **worker 负责生成**
- **frontend 负责最终协议整形**
- **Rust HTTP 层负责最终对外传输**
