# DEP-XXXX PR 2 详细分解：Transport 抽象 + Deterministic Clock

> **状态**：DRAFT v1.0 — 2026-04-20
> **依赖**：PR 1（需要 `PluginRegistry` proto + `PipelineContext` 做 contract test echo plugin）
> **下游**：PR 3（用 transport + clock 做 registry server / scheduler）、PR 5（用 clock 做 tick 调度）
> **预估工作量**：1 工程师 × 4-5 天；2 工程师并行 ~2.5-3 天

---

## 修订历史

### v2.1（2026-04-28）—— wire transport Pydantic↔proto bridge 漏装修复

**背景**：写第一个真外部 plugin 端到端测试 (`tests/integration/test_external_plugin_e2e.py`) 时发现 `_GrpcTransportBase.call()` **没有把 Pydantic stage request 转成 proto** 就直接交给 gRPC stub，导致 `Message.SerializeToString` 异常。这是 PR 2 ship 起就存在的 bug，因为：

- pipeline (PR 5+) emits Pydantic（`ProposeStageRequest` 等）才能继续用属性访问做 `_response_to_plugin_result`
- gRPC stub 只接受 proto message
- transport contract test 直接用 proto 做 round-trip → 漏过了"pipeline 真把 Pydantic 喂进 transport.call" 这条路径
- 所有 e2e 测试都走 `register_internal` → `InProcessTransport`，Pydantic 透传 → 也漏过

**修复（`plugins/transport/_grpc_base.py`）**：

```python
request_was_pyd = isinstance(request, BaseModel)
wire_request = pydantic_to_proto(request) if request_was_pyd else request
wire_response = await asyncio.wait_for(rpc(wire_request), self.timeout_seconds)
if request_was_pyd and isinstance(wire_response, ProtoMessage):
    return proto_to_pydantic(wire_response)
return wire_response
```

对称语义："Pydantic in → Pydantic out"（pipeline 路径）；"proto in → proto out"（transport contract test 路径，保持 byte-equal round-trip 不变）。两侧都受 `_proto_bridge.py` 既有 round-trip 测试覆盖。

**回归状态**（修复后）：
- transport contract 50/50（proto 路径）✅
- 新加 5/5 external plugin e2e（Pydantic 路径，UDS + grpc，单插件 + 双插件 priority merge + unregister）✅
- dual-path parity 30/30 ✅（不动决策输出）
- transport + registry + orchestrator + integration 总 305/305 ✅

**影响**：PR 5 文档说"orchestrator 通过 transport.call 调插件" 现在**真实可工作**；之前只在 in-process 路径下证明过。

### v2.0（2026-04-22）—— PR 2 实施完成 + 偏差同步

**实施状态**：全部 8 sub-task 编码完成；88 个测试全过（in_process 9 + clock 9 + transport_contract 50 + config 13 + mtls 7）；contract test 含 32 个 transport×input round-trip + 8 个 byte-equal 跨 transport 验证。

**与 v1.x 文档实施偏差**（按发现顺序）：

| 偏差点 | v1 文档 | 实际实施 | 决策理由 |
|---|---|---|---|
| 文件命名 | `transport/grpc.py` | `transport/grpc_remote.py` | 避免与标准库 `grpc` 包同名（python import shadow） |
| `_grpc_base.py` 抽出 | 隐含在 uds.py / grpc.py 各自实现 | 独立 mixin `_GrpcTransportBase` 共享 call/close/error mapping | uds 与 grpc 的 dispatch + error mapping 完全相同；只有 `_build_channel()` 不同 |
| `_method_dispatch.py` 抽出 | 隐含在 transport 内 | 独立 `StubDispatcher` 类 | 让 method 名 → stub 映射可单独单测；channel-per-plugin + lazy stub cache 模式清晰 |
| `MtlsConfig` 验证 | 仅 PEM 头检查 | 加 empty-file 显式拒绝 + 含 `-----BEGIN` 头检查 | 测试时 empty file 行为容易出现 |
| gRPC `aio.AioRpcError` 映射 | 简单 wrap PluginCallError | 详细映射：`UNAVAILABLE→Connection`、`UNIMPLEMENTED→UnknownMethod`、`INTERNAL/DATA_LOSS→Serialization` | 给 PR 5 circuit breaker 提供更精准的 retry/backoff 信号 |
| `executor_max_workers` 上限 | 标 follow-up | 仍标 follow-up（PR 7 production config 时引入）| sync plugin 红线问题在 README 提示足够 v1 |

**v2.0 实施补全**：

- `_method_dispatch.py` —— 共享方法 dispatch 表 + StubDispatcher，PR 3 `register_internal` 路径会复用
- `_grpc_base.py` —— uds + grpc 共享基类；任何 gRPC 错误的统一映射在此一处
- `transport/config.py` —— 完整 Pydantic schema + `make_transport_for_endpoint` + `make_clock`；后者带 production safety check（`DYNAMO_PLANNER_TEST=1` env override）
- contract test 含 self-signed cert 自动生成（`cryptography` 包，dynamo 已有），mTLS 路径无需额外 fixture
- README + Threat Model 详细描述 3 transport 各自信任假设与缓解

**与下游 PR 的契约**：

- `PluginTransport` ABC、`InProcessTransport` 直接被 PR 5 orchestrator 用
- `make_transport_for_endpoint(plugin_id, endpoint, config, in_process_instance=...)` 是 PR 3 `register_internal` 与 `register` 共用的 transport factory（P1-5 review v11 明确）
- `Clock` 抽象给 PR 3 HeartbeatMonitor / CircuitBreaker / PluginScheduler 用

### v1.0（2026-04-20）—— 初稿
- 与主文档 v10、Implementation Breakdown 对齐
- 8 个 sub-task：transport 抽象 / 3 种 transport 实现 / clock 抽象 / contract test / mTLS 配置 / 文档
- **复用 dynamo 平台级 cert-manager 基础设施**（不在 PR 2 引入新 PKI）
- planner 已有 `tick_input.now_s` 时间注入习惯——`Clock` 抽象只是把它形式化，零业务影响

---

## 为什么 PR 2 风险中等

| 维度 | 评级 | 理由 |
|---|---|---|
| 与现有代码冲突 | **极低** | 全部新建文件 |
| Transport contract 设计正确性 | **中** | 3 种 transport 必须输出位级一致——错了会让 PR 5/6/7 难调试 |
| gRPC mTLS 集成复杂度 | **中-高** | grpc.aio + mTLS + cert reload 是 boilerplate-heavy；建议**复用 dynamo 平台已有 cert-manager 配置约定**，不引入新 PKI |
| Clock 抽象设计 | **低** | dynamo planner 已有 `now_s` 注入习惯（参考 `tick_input.now_s`）；PR 2 只是把它形式化为 `Clock` interface |
| 阻塞下游 | **高** | PR 3 / PR 5 都依赖 |

**核心风险 1**：3 种 transport 输出不一致 → 调试困难。**唯一缓解**：1-3 sub-task contract test 必须严格 byte-equality。

**核心风险 2**：mTLS 配置耦合 dynamo K8s deployment → 在 dev / 单元测试场景下不可用。**缓解**：mTLS 仅在 `grpc` transport 启用；in_process / uds 完全不涉及；CI 测试用 `static_secret` fallback。

---

## 范围

**新建**：
- `components/src/dynamo/planner/plugins/transport/` 目录（base + 3 种 transport 实现）
- `components/src/dynamo/planner/plugins/clock.py`（Clock + 2 种实现）
- 配置 schema（mTLS / clock 类型 toggle）
- contract test + 单测

**不动**：
- 不动 `core/state_machine.py` 的 `tick_input.now_s` 现状（PR 7 才接入 Clock）
- 不实现 `PluginRegistry` server / `PluginScheduler`（PR 3）
- 不引入新的 K8s 资源 / RBAC / cert-manager 配置（复用 dynamo 现有平台基础设施）

---

## 子任务清单（8 项）

### 2-1：Transport ABC + 共用错误类型

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/__init__.py`<br/>`components/src/dynamo/planner/plugins/transport/base.py`<br/>`components/src/dynamo/planner/plugins/transport/errors.py` |
| 接口 | <pre>class PluginTransport(abc.ABC):<br/>    plugin_id: str<br/>    endpoint: str  # inproc:// / unix:// / grpc://<br/>    timeout_seconds: float  # per-RPC timeout<br/><br/>    @abc.abstractmethod<br/>    async def call(self, method: str, request: Any) -> Any: ...<br/>    """Call any plugin RPC method by name. Returns raw response message.<br/>       Method names: 'Predict' / 'Propose' / 'Reconcile' / 'Constrain'<br/>                     / 'Bootstrap' / 'Reset'.<br/>       Throws PluginCallError on any failure."""<br/><br/>    @abc.abstractmethod<br/>    async def close(self) -> None: ...<br/><br/>class PluginCallError(Exception):<br/>    pass<br/>class PluginTimeoutError(PluginCallError): pass<br/>class PluginConnectionError(PluginCallError): pass<br/>class PluginSerializationError(PluginCallError): pass<br/>class PluginUnknownMethodError(PluginCallError): pass</pre> |
| **关键设计决策** | <ol><li>**统一 `call(method, request)` 接口**——而非每个 RPC 一个 method（如 `predict()` / `propose()`）。理由：让 orchestrator 的流水线驱动代码（PR 5）一视同仁处理所有 stage，可以用 `for method, plugin in active_set: await plugin.transport.call(method, ctx)` 这样的统一循环</li><li>**`close()` 必须 idempotent**——orchestrator shutdown 时可能多次调（双重 cleanup 安全）</li><li>**所有失败抛 `PluginCallError` 子类**——不允许漏 catch。subtype 帮助上层选择性重试 / circuit breaker 触发</li></ol> |
| 单测 | `tests/plugins/transport/test_base.py`：mock 子类，验证 `PluginCallError` 继承链 |
| 依赖 | PR 1 的 `proto_gen` |
| 估算 | 0.5 天 |

---

### 2-2：InProcessTransport

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/in_process.py` |
| 接口 | <pre>class InProcessTransport(PluginTransport):<br/>    def __init__(self, plugin_id, instance, timeout_seconds=5.0):<br/>        self.plugin_id = plugin_id<br/>        self.endpoint = f"inproc://{plugin_id}"<br/>        self.timeout_seconds = timeout_seconds<br/>        self._instance = instance  # the plugin Python object<br/><br/>    async def call(self, method: str, request):<br/>        fn = getattr(self._instance, method, None)<br/>        if fn is None:<br/>            raise PluginUnknownMethodError(f"{method} not on {self.plugin_id}")<br/>        try:<br/>            coro = fn(request) if asyncio.iscoroutinefunction(fn) \<br/>                   else asyncio.to_thread(fn, request)<br/>            return await asyncio.wait_for(coro, self.timeout_seconds)<br/>        except asyncio.TimeoutError:<br/>            raise PluginTimeoutError(...)<br/>        except Exception as e:<br/>            raise PluginCallError(...) from e<br/><br/>    async def close(self):<br/>        # in-process plugin lifecycle owned by orchestrator<br/>        pass</pre> |
| **关键设计决策** | <ol><li>**支持 sync 与 async plugin 两种**——`asyncio.iscoroutinefunction()` 检测；sync 用 `asyncio.to_thread` 跑（避免阻塞 event loop）。注：builtin plugin 都是 async（PR 6 约定），sync 路径主要给 in_process user plugin（PR 5 5-6 的 in_process_loader）</li><li>**timeout 用 `asyncio.wait_for`** ——和 grpc transport 行为一致（per-RPC timeout 而非 connection timeout）</li><li>**不**做参数序列化——in_process 直接传 Python 对象（Pydantic class），零开销。这是 in_process 相对 uds/grpc 的核心优势</li><li>**没有 close 内容**——plugin instance 生命周期由 orchestrator 管（与 uds/grpc 不同；那两个有 channel/socket 要 close）</li></ol> |
| 单测 | `tests/plugins/transport/test_in_process.py`：<br/>- async plugin 调用成功<br/>- sync plugin 调用成功（验证 `asyncio.to_thread` 生效）<br/>- `PluginTimeoutError` 触发<br/>- `PluginUnknownMethodError`（method 不存在）<br/>- 异常被 wrap 为 `PluginCallError` |
| 依赖 | 2-1 |
| 估算 | 0.5 天 |

---

### 2-3：UdsTransport（grpc-over-uds）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/uds.py` |
| 接口 | <pre>class UdsTransport(PluginTransport):<br/>    def __init__(self, plugin_id, endpoint, timeout_seconds=5.0):<br/>        self.plugin_id = plugin_id<br/>        self.endpoint = endpoint  # unix:///path/to/socket<br/>        self.timeout_seconds = timeout_seconds<br/>        self._channel = None  # lazy init<br/>        self._stubs = {}      # method -> stub<br/><br/>    async def call(self, method: str, request):<br/>        if self._channel is None:<br/>            self._channel = grpc.aio.secure_channel(...) \<br/>                if self._wants_tls() else grpc.aio.insecure_channel(<br/>                    f"unix:{self._path}", options=...)<br/>        stub = self._get_stub(method)  # picks the right service stub<br/>        try:<br/>            return await asyncio.wait_for(<br/>                stub(request), self.timeout_seconds)<br/>        except grpc.aio.AioRpcError as e:<br/>            raise PluginCallError(...) from e<br/>        ...</pre> |
| **关键设计决策** | <ol><li>**uds 不强制 mTLS**——同 Pod sidecar 信任边界由 Pod boundary 提供；TLS 增加 CPU 开销无收益。配置上**禁止**在 `unix://` endpoint 配 mTLS（2-7 schema 校验）</li><li>**channel 复用**——每个 plugin 一个 channel；method-level stub cache。channel-per-plugin 而非 channel-per-RPC（gRPC 习惯）</li><li>**method 名 → stub 映射**：在 transport 内维护一张 `{ "Predict": stub.PredictPlugin.Predict, "Bootstrap": stub.PluginLifecycle.Bootstrap, ... }` 表；按 method 名 dispatch；method 不存在抛 `PluginUnknownMethodError`</li><li>**socket 文件不存在**抛 `PluginConnectionError`——orchestrator 上层据此触发 circuit breaker</li><li>**socket path 长度限制**：UDS path Linux 上 108 chars 限制；2-3 单测必须覆盖（绝对路径过长场景）</li></ol> |
| 单测 | `tests/plugins/transport/test_uds.py`：<br/>- 启 grpc.aio server 在 tmp UDS path → echo plugin → call → 验证 round-trip<br/>- socket 不存在抛 `PluginConnectionError`<br/>- timeout 抛 `PluginTimeoutError`<br/>- channel 复用（连续 100 调用只 init 一次） |
| 依赖 | 2-1, PR 1（generated stub）|
| 估算 | 1 天 |

---

### 2-4：GrpcTransport（含 mTLS 配置加载）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/grpc.py`<br/>`components/src/dynamo/planner/plugins/transport/_mtls.py`（mTLS cred 加载工具）|
| 接口 | <pre>class GrpcTransport(PluginTransport):<br/>    def __init__(self, plugin_id, endpoint, mtls_config=None, timeout_seconds=5.0):<br/>        self.plugin_id = plugin_id<br/>        self.endpoint = endpoint  # grpc://host:port<br/>        self.timeout_seconds = timeout_seconds<br/>        self._mtls = mtls_config  # MtlsConfig or None<br/>        self._channel = None<br/><br/>    async def call(self, method, request):<br/>        if self._channel is None:<br/>            self._channel = self._build_channel()<br/>        # same dispatch as UdsTransport via stub map<br/>        ...<br/><br/>    def _build_channel(self):<br/>        target = self.endpoint.removeprefix("grpc://")<br/>        if self._mtls:<br/>            creds = grpc.ssl_channel_credentials(<br/>                root_certificates=self._mtls.ca_bundle,<br/>                private_key=self._mtls.client_key,<br/>                certificate_chain=self._mtls.client_cert)<br/>            return grpc.aio.secure_channel(target, creds, options=...)<br/>        return grpc.aio.insecure_channel(target, options=...)</pre> |
| `MtlsConfig` 加载 | <pre>@dataclass<br/>class MtlsConfig:<br/>    ca_bundle: bytes<br/>    client_cert: bytes<br/>    client_key: bytes<br/><br/>    @classmethod<br/>    def from_files(cls, ca_path, cert_path, key_path) -> "MtlsConfig":<br/>        ...  # read files, validate PEM<br/><br/>    @classmethod<br/>    def from_k8s_secret_mount(cls, mount_dir: str) -> "MtlsConfig":<br/>        # convention: $mount/ca.crt, $mount/tls.crt, $mount/tls.key<br/>        # matches dynamo platform cert-manager / certificateSecret 习惯<br/>        return cls.from_files(...)</pre> |
| **关键设计决策** | <ol><li>**复用 dynamo 平台 cert-manager 约定**——mount path 与 `deploy/helm/charts/platform/values.yaml` 中 `certificateSecret`（`tls.crt` / `tls.key` / `ca.crt` 三键）一致。**不**引入新的 K8s 资源 / 新的 mount path / 新的 RBAC</li><li>**`grpc://` endpoint 强制 mTLS（默认）**；如需明文（如 dev），通过配置 `planner.plugin_registration.transport.allow_insecure_grpc: true` 显式开启 + 启动时 WARNING log</li><li>**cert reload**：v1 **不**实现（cert-manager 默认会触发 Pod restart 完成轮换）；v2 follow-up 加 `inotify` 监听 cert mount 自动重连</li><li>**channel options**：keepalive / max_message_size 在 `_GRPC_CHANNEL_OPTIONS` 常量中集中配；不让单个 plugin 独立改</li></ol> |
| 单测 | `tests/plugins/transport/test_grpc.py`：<br/>- 启本地 grpc.aio server (insecure) → echo → call → 验证 round-trip<br/>- 启本地 grpc.aio server (mTLS, 用 self-signed test cert) → mTLS handshake 成功<br/>- cert path 不存在抛 `PluginConnectionError`<br/>- 错误的 cert（如 server cert 给 client）抛 `PluginConnectionError`<br/>- timeout 抛 `PluginTimeoutError`<br/>`tests/plugins/transport/test_mtls.py`：<br/>- `from_files` 成功 / 失败<br/>- `from_k8s_secret_mount` 成功 / 失败<br/>- PEM 校验（坏 cert 抛 `ValueError`） |
| 依赖 | 2-1, PR 1（generated stub）|
| 估算 | 1.5 天 |

---

### 2-5：Clock 抽象 + WallClock / VirtualClock

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/clock.py`<br/>`tests/plugins/clock/test_wall_clock.py`<br/>`tests/plugins/clock/test_virtual_clock.py` |
| 接口 | <pre>class Clock(abc.ABC):<br/>    @abc.abstractmethod<br/>    def now(self) -> float:<br/>        """Wall-clock seconds (epoch float)."""<br/>    <br/>    @abc.abstractmethod<br/>    def monotonic(self) -> float:<br/>        """Monotonic seconds (for measuring durations)."""<br/>    <br/>    @abc.abstractmethod<br/>    async def sleep(self, seconds: float) -> None: ...<br/><br/>class WallClock(Clock):<br/>    def now(self): return time.time()<br/>    def monotonic(self): return time.monotonic()<br/>    async def sleep(self, s): await asyncio.sleep(s)<br/><br/>class VirtualClock(Clock):<br/>    def __init__(self, start_now=0.0, start_mono=0.0):<br/>        self._now = start_now<br/>        self._mono = start_mono<br/>        self._sleepers = []  # heapq of (wake_at, future)<br/><br/>    def now(self): return self._now<br/>    def monotonic(self): return self._mono<br/>    async def sleep(self, s):<br/>        fut = asyncio.get_event_loop().create_future()<br/>        heapq.heappush(self._sleepers, (self._mono + s, fut))<br/>        await fut<br/><br/>    def advance(self, seconds: float):<br/>        self._now += seconds; self._mono += seconds<br/>        while self._sleepers and self._sleepers[0][0] <= self._mono:<br/>            _, fut = heapq.heappop(self._sleepers)<br/>            if not fut.done(): fut.set_result(None)</pre> |
| **关键设计决策** | <ol><li>**`now()` 与 `monotonic()` 区分**——`now()` 给 audit log / `tick_input.now_s`（epoch 时间）；`monotonic()` 给 duration / scheduling（不受 NTP / clock skew 影响）</li><li>**`VirtualClock.advance` 触发 sleep wake**——通过 heapq 维护 sleeping coroutines；`advance(N)` 把所有 wake_at <= now+N 的都唤醒；这让 PR 5 测试可以"瞬时跑完 1 小时"</li><li>**禁止全局 clock singleton**——orchestrator 持有 clock 实例并按依赖注入传给所有需要时间的组件（plugin / scheduler / metric collector）。**任何 `time.time()` 调用都是 bug**（`pytest` lint check 在 PR 5 5-9 加上）</li><li>**`monotonic` start 默认 0.0**——VirtualClock 测试时 timestamps 整数好读；不与 wall clock 同步</li></ol> |
| 单测 | <ol><li>`WallClock`：基本调用通；`now()` ≈ `time.time()`；`monotonic()` 单调递增；`sleep(0.1)` 真等 0.1s</li><li>`VirtualClock`：<br/>  - `advance(N)` → `now()` / `monotonic()` 同步加 N<br/>  - 多个 sleeper 按时间序唤醒<br/>  - `advance` 跳过 sleep deadline → 睡眠者立即唤醒<br/>  - 复杂场景：5 个 coroutine 各 sleep 不同时长 + 1 次 `advance(10)` → 全部唤醒且 `now()` = +10</li></ol> |
| 依赖 | 无（与 transport 完全独立，可 2-1 同时启动）|
| 估算 | 0.75 天 |

---

### 2-6：Transport contract test（核心 acceptance）

| 项 | 内容 |
|---|---|
| 实现位置 | `tests/plugins/transport/test_transport_contract.py` |
| 目的 | **3 种 transport 输出位级一致** —— 这是 PR 2 最重要 acceptance；任何分歧立刻暴露 |
| 接口 | <pre>@pytest.fixture(params=["in_process", "uds", "grpc", "grpc_mtls"])<br/>def echo_transport(request):<br/>    """Set up echo plugin reachable via the parametrized transport.<br/>       Returns ready-to-use PluginTransport."""<br/>    ...<br/><br/>@pytest.mark.parametrize("ctx_factory", [<br/>    _make_minimal_ctx,<br/>    _make_full_ctx,<br/>    _make_ctx_with_fpm,<br/>    _make_ctx_with_multi_target_proposal,<br/>    _make_ctx_with_unicode,<br/>])<br/>async def test_round_trip(echo_transport, ctx_factory):<br/>    ctx_in = ctx_factory()<br/>    response = await echo_transport.call("Predict",<br/>                  PredictStageRequest(context=ctx_in))<br/>    ctx_out = response.predictions  # echo plugin returns input as predictions<br/>    assert ctx_out.SerializeToString() == ctx_in.SerializeToString()</pre> |
| **关键测试场景**（5 ctx × 4 transport = 20 test cases）| <ol><li>minimal context：仅 request_id</li><li>full context：所有 6 字段填满</li><li>含 FpmData（`map<string, bytes>`，msgspec encoding）</li><li>多 component proposal（prefill+decode 一次）</li><li>含 unicode（reason="测试中文" + emoji）</li></ol> |
| 错误场景测试 | <ol><li>所有 4 transport：调不存在 method 抛 `PluginUnknownMethodError`</li><li>所有 4 transport：plugin 内部抛异常 → 调用方收到 `PluginCallError`</li><li>所有 4 transport：超时 → `PluginTimeoutError`</li><li>uds / grpc：endpoint unreachable → `PluginConnectionError`</li></ol> |
| pytest markers | `pytest.mark.{pre_merge, planner, gpu_0, unit}`（自动接现有 `planner-test` job）|
| 依赖 | 2-2, 2-3, 2-4, PR 1 |
| 估算 | 1 天（含 echo plugin 实现 + grpc server fixture） |

---

### 2-7：配置 schema + Pydantic 加载

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/config.py` |
| 接口 | <pre>class TransportConfig(BaseModel):<br/>    """planner.plugin_registration.transport 配置树"""<br/>    allow_insecure_grpc: bool = False  # default 拒绝明文 grpc<br/>    grpc_mtls: GrpcMtlsConfig \| None = None<br/>    request_timeout_seconds: float = 5.0<br/>    keepalive_time_ms: int = 30_000<br/>    max_message_size_bytes: int = 10_000_000  # 10 MB<br/><br/>class GrpcMtlsConfig(BaseModel):<br/>    enabled: bool = True<br/>    secret_mount_path: str = "/var/run/dynamo/planner-tls"  <br/>    # 与 dynamo platform 现有约定一致：tls.crt/tls.key/ca.crt 三键<br/>    cert_reload_inotify: bool = False  # v1 不实现，留 schema 字段<br/><br/>class ClockConfig(BaseModel):<br/>    """planner.scheduling.clock 配置树"""<br/>    type: Literal["wall", "virtual"] = "wall"<br/>    virtual_start_now: float = 0.0  # 仅 type=virtual 时用</pre> |
| **配置 → transport 工厂** | <pre>def make_transport_for_endpoint(<br/>    plugin_id: str, endpoint: str, config: TransportConfig,<br/>    in_process_instance=None,<br/>) -> PluginTransport:<br/>    if endpoint.startswith("inproc://"):<br/>        if in_process_instance is None:<br/>            raise ValueError("inproc endpoint needs instance")<br/>        return InProcessTransport(plugin_id, in_process_instance, config.request_timeout_seconds)<br/>    elif endpoint.startswith("unix://"):<br/>        return UdsTransport(plugin_id, endpoint, config.request_timeout_seconds)<br/>    elif endpoint.startswith("grpc://"):<br/>        if not config.allow_insecure_grpc and not config.grpc_mtls:<br/>            raise ValueError("grpc endpoint requires mTLS unless allow_insecure_grpc=true")<br/>        mtls = MtlsConfig.from_k8s_secret_mount(config.grpc_mtls.secret_mount_path) if config.grpc_mtls and config.grpc_mtls.enabled else None<br/>        return GrpcTransport(plugin_id, endpoint, mtls, config.request_timeout_seconds)<br/>    raise ValueError(f"unknown endpoint scheme: {endpoint}")</pre> |
| 单测 | `tests/plugins/transport/test_config.py`：<br/>- 4 种 endpoint scheme 各产出正确 transport 类<br/>- `grpc://` 无 mTLS + `allow_insecure_grpc=False` → ValueError<br/>- `unix://` + mTLS 配置 → 校验失败（mTLS 仅 grpc）<br/>- 默认配置（`allow_insecure_grpc=False`, `grpc_mtls.enabled=True`）符合「生产安全」预期 |
| 依赖 | 2-2, 2-3, 2-4 |
| 估算 | 0.5 天 |

---

### 2-8：README + Threat Model

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/transport/README.md` |
| 内容 | <ol><li>3 种 transport 选择决策树（in_process: 同进程 / uds: 同 Pod / grpc: 跨 Pod）</li><li>每种 transport 的部署 example（K8s Pod yaml 片段：sidecar / 独立 deployment）</li><li>mTLS 配置 quick start（用 dynamo platform cert-manager + secret mount 一键启用）</li><li>**Threat Model**：每 transport 攻击面 + 推荐缓解（in_process: 同进程 trust / uds: filesystem ACL / grpc: mTLS）</li><li>Clock 选择：production 用 `wall`；test / replay 用 `virtual`</li><li>故障排查：常见 `PluginCallError` 子类 → 排查步骤</li></ol> |
| 依赖 | 2-1 ~ 2-7 全部完成 |
| 估算 | 0.5 天 |

---

## PR 2 总估算

- **单人**：~6.25 天（2-1: 0.5 + 2-2: 0.5 + 2-3: 1 + 2-4: 1.5 + 2-5: 0.75 + 2-6: 1 + 2-7: 0.5 + 2-8: 0.5 = 6.25 天）
- **双人并行**：~3-3.5 天
  - 工程师 A：2-1 → 2-2 → 2-3 → 2-7（in_process / uds 链路 + 配置）
  - 工程师 B：2-5（独立）→ 2-4（grpc + mTLS）
  - 汇合：2-6（contract test）→ 2-8（README）

注：原 Implementation Breakdown 估算 1 工程师 × 4-5 天偏乐观。本 PR 2 详细文档建议按 **单人 6 天 / 双人 3 天** 排期，主要差异在 grpc + mTLS sub-task（2-4）+ contract test（2-6）实际工作量。

---

## PR 2 Acceptance Criteria

- [ ] `PluginTransport` ABC + 4 subclass（InProcess / Uds / Grpc / Grpc+mTLS）全部实现
- [ ] **Contract test 20 个 case 全部通过**（5 ctx × 4 transport，byte-equality）
- [ ] **错误场景测试 ~16 个 case 全部通过**（每 transport 的 timeout / unknown method / connection / wrap exception）
- [ ] `Clock` ABC + WallClock + VirtualClock 实现，单测全部通过
- [ ] `VirtualClock.advance` 触发 sleep wake 行为正确（多 sleeper / 跳过 deadline 等场景）
- [ ] mTLS 配置加载兼容 dynamo platform cert-manager 现有 secret mount 约定（`tls.crt` / `tls.key` / `ca.crt`）
- [ ] CI `planner-test` job 自动包含本 PR 全部测试（marker discovery）
- [ ] PR description 明确：本 PR **零 production 影响**（无业务调用、orchestrator 不存在、planner core 不动）
- [ ] PR description 警告：**禁止任何 production code 用 `time.time()` / `time.monotonic()` / `asyncio.sleep`**——必须用 `Clock`；linter check 在 PR 5 5-9 启用

---

## 跨 Sub-task 必须协调的点

### 1. `call(method, request)` 接口的统一性

3 种 transport 必须**同一签名同一行为**：
- 输入：`method: str, request: <proto message or pydantic model>`
- 输出：`response: <proto message or pydantic model>`
- 失败：抛 `PluginCallError` 子类

PR 5/6/7/8 写 plugin 调用代码时假定这条契约；任何 transport 偏离立刻把上层代码搞复杂。

### 2. `request` / `response` 类型选择：proto generated 还是 Pydantic？

**决议**：`call()` 接口**两端都用 proto generated message**（`PredictStageRequest` / `PredictStageResponse` 等）；上层（orchestrator）在调用前用 PR 1 的 `pydantic_to_proto` 转换。

理由：
- in_process transport 直接 Python 调用，要求 plugin signature 是 `def Predict(req: PredictStageRequest) -> PredictStageResponse`——这强制 in_process plugin 也用 proto class，**与 uds/grpc 一致**
- 否则 in_process 用 Pydantic、其他用 proto，3 transport 行为不一致，contract test 等于自欺欺人

**例外**：plugin 代码内部可以随意 Pydantic 转换处理；但 `call()` 边界严格 proto。

### 3. mTLS 配置只走 K8s secret mount，不走 in-line config

PR 2 **明确禁止**在 yaml 中写 `client_cert: |\n -----BEGIN CERTIFICATE-----` 这种 in-line cert——必须走 K8s secret mount。理由：
- 防止 cert 进入 git / config map（安全）
- 复用 dynamo platform cert-manager 自动 rotation 机制
- 简化 PR 2 schema（只需 `secret_mount_path` 一个字段）

`from_files` API 保留是为了**单元测试**用 self-signed test cert（`tests/_fixtures/test_cert/`）。

### 4. `VirtualClock` 只在测试 / replay 用，不在 production 配置允许

PR 5 5-9 加 `pytest.mark.parametrize` 让 orchestrator 测试可以选用 `WallClock` 或 `VirtualClock`。production NativePlannerBase 启动代码（PR 7）**强制**用 `WallClock`——配置 `clock.type=virtual` 在生产环境启动应抛 `ValueError("VirtualClock not allowed in production; set DYNAMO_PLANNER_TEST=1 to override")`。

---

## 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| grpc.aio + mTLS 集成 boilerplate 多 → 2-4 sub-task 实际超期 | **中** | 2-4 单独排 1.5 天预算；建议优先做 2-2 / 2-3 / 2-5 / 2-6（不依赖 mTLS 的部分）让 PR 3 / PR 5 早启动 |
| 3 种 transport contract 不一致 → 难调试 | **中** | 2-6 contract test 是 must-have；任何 transport 的新功能必须先加 contract case |
| Clock 抽象 leak → 业务代码偷偷调 `time.time()` | **中** | PR 5 5-9 加 `ast.parse` based linter，扫描 plugins 子模块禁止 `time.time` / `time.monotonic` / `asyncio.sleep`（直接调） |
| dynamo cert-manager 约定与 PR 2 假设不符 | **低-中** | 2-4 sub-task 启动前**先 review** `deploy/helm/charts/platform/values.yaml` 中 `certificateSecret` 配置；如发现 mount path / 文件名不一致，**改 PR 2 适应平台**（不要反过来） |
| sync plugin（in_process）`asyncio.to_thread` 跑过慢 | **低** | builtin plugin 全部 async（PR 6 约定）；in_process user plugin 大概率轻量（regression query 等）；如发现性能问题，加 `concurrent.futures.ProcessPoolExecutor` 不在 PR 2 范围 |

---

## 推荐 staffing

- **1 名 Backend 工程师**（gRPC / asyncio 熟练）：负责 2-1 / 2-3 / 2-4 / 2-7
- **1 名 Backend / Infra 工程师**（K8s / cert-manager 熟练）：负责 2-2 / 2-5 / 2-6 / 2-8

---

## Resolved Questions（已决议）

### Q1：`call()` 是否暴露 cancellation token？

**决议**：**不**。理由：
- `asyncio.wait_for` 内部已处理 timeout cancellation；调用方不需要手动管理 cancellation
- orchestrator 整 tick 用 `asyncio.wait_for(timeout=tick_max_duration_seconds)` 做兜底（PR 5 5-4 实现），单 plugin cancellation 自动 propagate
- 暴露 token 增加 API 复杂度但没用例

### Q2：`InProcessTransport` 是否要做 deepcopy 隔离？

**决议**：**不**。理由：
- in_process plugin 与 orchestrator 在同一 event loop / 同一进程；plugin 修改入参 = 接受这个不变量
- 如果 plugin 真的需要保护，自己在 plugin 内 deepcopy（不强制 transport 做）
- deepcopy 大 PipelineContext（含 FpmData bytes）开销显著

但**约定**（在 README 中）：plugin 实现**禁止**修改入参——plugins should treat request as immutable。这是社会契约，不是 transport 强制。

### Q3：grpc keepalive / max_message_size 是否暴露给 plugin 配？

**决议**：**不**。理由：
- 每 plugin 单独配 → orchestrator 难统一限流 / 难审计
- 集中在 `TransportConfig`（PR 2-7）；如某 plugin 需要更大 message size，**先讨论是否改架构**（如分批传 FPM）

### Q4：是否需要 `PluginTransportPool`（连接池）？

**决议**：**不**（v1）。理由：
- 当前模型每 plugin 独立 channel（gRPC channel 自带 connection multiplexing）；不需要 transport-level pool
- 如果未来 plugin 数量极大（>100），加 pool 是优化不是 prerequisite

### Q5：是否在 PR 2 实现 cert hot reload？

**决议**：**不**（v1）。理由：
- cert-manager 默认 cert 临到期前几小时 trigger Pod restart 完成轮换——足够
- inotify-based hot reload 复杂度高，错了会让 grpc connection 中断
- v2 follow-up 加（schema 字段 `cert_reload_inotify` 已预留）

---

## 已删除的内容（v0 → v1）

无（首版）

---

## 与其他 PR 的接口

| 下游 PR | 依赖 PR 2 的内容 |
|---|---|
| PR 3（Registry + Scheduler） | `PluginTransport` / `make_transport_for_endpoint`：Register 时为新 plugin 创建 transport instance；scheduler 用 `Clock.now()` 算 active set；`Clock.monotonic()` 算 cache age |
| PR 5（Orchestrator） | `Clock` 注入到 orchestrator；`PluginTransport.call()` 在 pipeline 中调每个 plugin；`VirtualClock` 在测试中推进时间 |
| PR 7（NativePlannerBase 双路径） | 启动代码用 `WallClock` 实例化 orchestrator |
| PR 8（Replay） | `ReplayPlannerAdapter` 用 `VirtualClock` 重放历史 trace |

---

## 下一步

1. **review 本 PR 2 详细文档**（重点：sub-task 拆分、契约统一性、mTLS 与 dynamo 平台衔接、Q1-Q5 决议）
2. **review 后启动 PR 2 实施**（建议双人并行：A 走 2-1/2-3/2-4/2-7；B 走 2-2/2-5/2-6/2-8）
