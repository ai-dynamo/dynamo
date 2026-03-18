# 从 Operator 过渡到 Runtime：Dynamo 请求链路学习计划（第二版）

这份第二版学习计划，不再重复第一版里“`agg.yaml -> DGD -> Operator -> Pod`”那条主线。

你现在已经基本完成了这部分理解：

- 知道 `DynamoGraphDeployment` 如何被 operator 接住
- 知道 operator 如何选择 Grove / DCD / LWS 路径
- 知道最终 Pod 里启动的是 `dynamo.frontend` 和 `dynamo.sglang`
- 也知道 Grove、Kai-Scheduler、Volcano 这些组件分别处在什么层

所以从现在开始，学习重点应该明显调整：

- 从“控制面如何把 YAML 变成 Pod”
- 切换到“数据面里一个请求是如何被 frontend 接住、路由、发给 worker、再流式返回”

这份计划的目标，是帮你建立一条新的主线：

```text
Client Request
  -> dynamo.frontend
  -> pre/post process
  -> router
  -> runtime endpoint / discovery
  -> dynamo.sglang worker
  -> engine.async_generate()
  -> stream back to frontend
  -> HTTP/SSE 返回给客户端
```

如果第一版的目标是：

- 看懂部署链路

那么第二版的目标就是：

- 看懂请求链路

---

## 一、先调整你的学习重心

你后面最值得投入时间的，不再是继续深挖 DGD controller 细节，而是下面这 4 层：

1. `frontend`
   - 请求从 HTTP/OpenAI API 进来时，第一站发生了什么
2. `processor/router`
   - frontend 怎么把请求变成内部生成请求，并决定发给哪个 endpoint
3. `runtime/discovery`
   - frontend 是怎么拿到 worker client 的，namespace.component.endpoint 又是怎么解析的
4. `sglang worker`
   - worker 收到请求以后，怎么调用 SGLang engine，再怎么把结果流回来

后面的学习顺序，我建议就围绕这 4 层走，而不是再从 operator 往下追。

原因很简单：

- 你已经理解了“Pod 是怎么来的”
- 现在真正缺的是“Pod 起来以后在干什么”

---

## 二、你现在最适合的新主线

围绕 `examples/backends/sglang/deploy/agg.yaml` 这个聚合例子，第二阶段最推荐的新问题是：

### 核心问题

一个客户端发到 `/v1/chat/completions` 的请求：

1. 是被 frontend 哪个入口接住的？
2. frontend 什么时候做 pre-process？
3. frontend 什么时候决定目标 worker？
4. frontend 是怎么找到 `decode.generate` 这个 endpoint 的？
5. sglang worker 收到请求后，调用的是哪段 handler？
6. handler 最终怎么调用 `engine.async_generate()`？
7. 结果是如何一段段流回 frontend 的？
8. frontend 怎么再把它包装成 OpenAI SSE / HTTP 响应？

你后面的阅读，建议都围绕这 8 个问题。

---

## 三、推荐的新学习顺序

下面这个顺序，是我根据你现在的理解阶段专门调整过的。

### 第 1 站：先读 `dynamo.frontend` 的总入口

源码入口：

- `components/src/dynamo/frontend/__main__.py`
- `components/src/dynamo/frontend/main.py`

这一章的目标不是抠细节，而是先建立框架感：

- frontend 进程启动时做了哪些初始化
- 它怎么解析参数
- 它什么时候创建 `DistributedRuntime`
- 它怎么选择 `router_mode`
- 它怎么决定用 `vllm_processor` 还是 `sglang_processor`

你先重点看：

- `parse_args()`
- `async_main()`

这一层你要先回答：

- frontend 这个进程本质上是“HTTP server + pre/post processor + router + runtime client”的组合体吗？

答案基本就是：是。

### 第 2 站：聚焦 `sglang_processor`

源码入口：

- `components/src/dynamo/frontend/sglang_processor.py`

这是你接下来最关键的一层。

因为在 `agg.yaml` 这个例子里，后端是 SGLang，所以 frontend 最值得读的是：

- `SglangEngineFactory`
- `_generate_and_stream(...)`

你要重点盯住下面几个动作：

1. 请求在这里被变成了什么内部对象
2. 这里是如何调用 `self.router.generate(...)` 的
3. 返回的是怎样的流
4. stream 中每个 chunk 是怎么被包装的

如果你只选一个文件作为“第二阶段真正起点”，我建议就是：

- `components/src/dynamo/frontend/sglang_processor.py`

### 第 3 站：把 frontend 的“路由”看明白

你在 frontend 里看见：

- `router.generate(...)`

之后，不要立刻跳很远，先问清楚：

- 这个 router 到底是 round-robin、direct 还是 kv router？

在 `frontend/main.py` 里你已经能看到：

- `router_mode == kv / random / direct / round-robin`

所以这一章的目标是：

- 先搞清楚 `agg.yaml` 这个例子实际更接近哪种 router 模式
- 再决定你下一步该追 `frontend` 内部 router 封装，还是追独立的 `dynamo.router`

建议先从 frontend 里“它怎么拿到 router client”看起，不要一开始就陷进 KV router 算法细节。

### 第 4 站：开始看 runtime 和 endpoint 解析

源码入口：

- `lib/runtime/src/distributed.rs`

这一层特别重要，因为它回答的是：

- frontend 为什么能写 `runtime.endpoint(...)`
- worker 为什么能 `serve_endpoint(...)`
- discovery backend 到底是在这层怎么接入的

你现在要把下面几件事串起来：

1. `DistributedRuntime` 初始化时怎么选择 discovery backend
2. Kubernetes discovery 和 KV store discovery 在 runtime 里怎么分流
3. `endpoint(namespace.component.endpoint)` 背后到底创建了什么抽象
4. frontend client 和 worker server 是怎么通过同一个 endpoint 名字对上的

这一层是你从“应用代码”进入“Dynamo runtime 抽象”的关键桥梁。

### 第 5 站：切到 `dynamo.sglang` 进程入口

源码入口：

- `components/src/dynamo/sglang/__main__.py`
- `components/src/dynamo/sglang/main.py`

这一章重点看：

- worker 进程启动时怎么解析配置
- 怎么创建 runtime
- 怎么根据 serving mode 分流到 `init_decode()` / `init_prefill()`

对于 `agg.yaml` 这个聚合例子，你最该重点追的是：

- `init_decode(...)`

因为单机聚合的 decode worker 多半就是这条线。

### 第 6 站：盯住 `init_decode(...)` 到 `serve_endpoint(...)`

源码入口：

- `components/src/dynamo/sglang/init_llm.py`

你要重点看这里的几件事：

1. worker 暴露的 endpoint 名字是什么
2. 是怎么 `runtime.endpoint(...)` 的
3. 是怎么 `serve_endpoint(handler.generate, ...)` 的
4. handler 是哪一个

这一步非常关键，因为它会把 frontend 那边的：

- `runtime.endpoint(...).client()`

和 worker 这边的：

- `serve_endpoint(...)`

真正接起来。

这一步看懂后，你就会知道：

- frontend 发请求，其实不是直接“找 Pod IP”
- 而是在走 Dynamo runtime 的 endpoint 抽象

### 第 7 站：精读 decode handler

源码入口：

- `components/src/dynamo/sglang/request_handlers/llm/decode_handler.py`

这是第二阶段最值得精读的 worker 文件之一。

你要重点看：

- `generate(...)`

并追这几个问题：

1. request 进入 handler 后先做了什么
2. 什么情况下走 aggregated
3. 什么情况下走 disaggregated
4. 最终是怎么调用：
   - `engine.async_generate(...)`
5. worker 返回的 stream item 长什么样
6. 哪些字段是 frontend 最终会继续转成 OpenAI chunk 的

如果你想真正知道“请求到 worker 后发生了啥”，这一层必须精读。

### 第 8 站：回头补 frontend 的返回链路

当你看完 worker handler 以后，再回到：

- `components/src/dynamo/frontend/sglang_processor.py`

重新看：

- `_generate_and_stream(...)`

这时你会更容易理解：

- frontend 收到 worker 的流式输出后
- 是怎么变成 OpenAI 风格 chunk 的
- 哪些 stop_reason / finish_reason / usage 字段是在这里处理的

这一站建议和第 7 站来回对照着看。

---

## 四、你后面最推荐的章节安排

我建议你把第二阶段拆成下面 6 篇。

### 第 1 篇：Frontend 进程启动全貌

学习目标：

- 看懂 `dynamo.frontend` 启动后创建了哪些核心对象
- 建立 `HTTP server + processor + router + runtime` 的整体认知

建议源码：

- `components/src/dynamo/frontend/main.py`
- `components/src/dynamo/frontend/frontend_args.py`

### 第 2 篇：SGLang Frontend 如何发起一次生成请求

学习目标：

- 看懂 `sglang_processor` 如何把一个 OpenAI 请求变成 Dynamo 内部生成调用

建议源码：

- `components/src/dynamo/frontend/sglang_processor.py`

重点函数：

- `_generate_and_stream(...)`

### 第 3 篇：Dynamo Runtime 的 endpoint / discovery 抽象

学习目标：

- 看懂 frontend 和 worker 为什么能通过 endpoint 名字互相找到

建议源码：

- `lib/runtime/src/distributed.rs`

### 第 4 篇：SGLang Worker 如何注册并暴露 generate endpoint

学习目标：

- 看懂 worker 怎么启动、怎么注册、怎么 serve endpoint

建议源码：

- `components/src/dynamo/sglang/main.py`
- `components/src/dynamo/sglang/init_llm.py`

### 第 5 篇：Decode Handler 如何调用 engine.async_generate()

学习目标：

- 看懂 worker 真正执行推理时的 handler 链路

建议源码：

- `components/src/dynamo/sglang/request_handlers/llm/decode_handler.py`
- `components/src/dynamo/sglang/request_handlers/handler_base.py`

### 第 6 篇：从 worker stream 回到 OpenAI SSE 的回包链路

学习目标：

- 看懂 frontend 如何把 worker 输出重新包装成客户端最终看到的流

建议源码：

- `components/src/dynamo/frontend/sglang_processor.py`

---

## 五、下一步最值得你立刻开始的文件

如果你现在就准备开始第二阶段，我建议你按这个顺序直接读：

1. `components/src/dynamo/frontend/main.py`
2. `components/src/dynamo/frontend/sglang_processor.py`
3. `lib/runtime/src/distributed.rs`
4. `components/src/dynamo/sglang/init_llm.py`
5. `components/src/dynamo/sglang/request_handlers/llm/decode_handler.py`

这个顺序的好处是：

- 先从入口看整体
- 再追请求发起
- 再补 runtime 抽象
- 最后进入 worker 执行

这样你不会一开始就被 worker 内部细节淹没。

---

## 六、和你当前进度最匹配的具体建议

你现在已经完成了 operator 这条线，所以我建议你后面遵守一个原则：

- 不要再继续横向扩展太多控制面内容
- 开始沿着“一次请求”的纵向链路，做从 frontend 到 worker 的穿透式阅读

也就是说，后面每次看代码都尽量问自己：

- “这个函数处在请求链路的哪一站？”

而不是只问：

- “这个函数本身在做什么？”

这是你从“看懂模块”走向“看懂系统行为”的关键一步。

---

## 七、我对你下一阶段的建议结论

一句话总结：

你下一阶段最应该学习的，不是继续挖 operator，而是：

- **以 `dynamo.frontend` 为入口，沿着一次 `/v1/chat/completions` 请求，追到 `sglang decode handler -> engine.async_generate()`，再追回前端 SSE 回包。**

如果要再压缩成一个最小起点，就是：

- 先读 `components/src/dynamo/frontend/main.py`
- 然后直接精读 `components/src/dynamo/frontend/sglang_processor.py`

这会是你现在最有收益的下一步。
