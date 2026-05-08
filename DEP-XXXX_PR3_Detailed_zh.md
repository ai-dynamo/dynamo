# DEP-XXXX PR 3 详细分解：PluginRegistry + 调度器

> **状态**：v2.1 — **全部 11 sub-task CODED**（v2.0 的 9 个 + PR 3.5 follow-up 的 2 个 / 2026-04-23）
> **依赖**：PR 1（proto + Pydantic types）+ PR 2（Transport + Clock）
> **下游**：PR 5（orchestrator 用 registry + scheduler 决定每 tick 调谁）
> **预估工作量**：1 工程师 × 1.5-2 周；2 工程师并行 ~7-9 天

---

## 修订历史

### v2.1（2026-04-23）—— PR 3.5 follow-up：K8sSATokenAuth + SpiffeJwtAuth 落地

本 session 把 PR 3 v1 留的两个 deferred auth source 实现完成并 wire 进 `build_auth_validator`。

**`plugins/registry/auth/k8s_sa_token.py`（新文件）**：
- `K8sSATokenAuth(kube_client, audience, trusted_service_accounts)` —— 接受 `kubernetes.client.AuthenticationV1Api` 或等价 ducktype 对象（DI 友好，测试可注入 fake）
- `validate(token)` 流程：
  1. `create_token_review(V1TokenReview(spec=V1TokenReviewSpec(token=, audiences=[audience])))` via `asyncio.to_thread`（sync K8s 客户端不阻塞 orchestrator 事件循环）
  2. 如果 `status.authenticated=False` → `AuthError`
  3. **Defence in depth**：`status.audiences` 必须回显我们请求的 audience —— API server 在 v1.20+ 会 filter，但 defensive check 防止旧集群或配置漂移
  4. 解析 `status.user.username` = `system:serviceaccount:<ns>:<sa>` 格式（真实 K8s SA username 用冒号分隔），normalise 为 `<ns>/<sa>`
  5. allow-list check
- API error（TokenReview 返 5xx 等）统一 catch + log + reraise AuthError，**不泄露 error body**（可能含 request ID / policy fragment）

**`plugins/registry/auth/spiffe_jwt.py`（新文件）**：
- `SpiffeJwtAuth(jwks_endpoint, audience, trust_domain, trusted_spiffe_ids, *, algorithms=None, jwk_client=None)` —— `jwk_client` 可注入（测试跳 HTTP fetch）；默认 alg 集合为 `RS256 + ES256`（SPIRE 实际签发的两个算法）
- `validate(token)` 流程：
  1. `PyJWKClient.get_signing_key_from_jwt(token)` 拉对应 `kid` 的 public key
  2. `jwt.decode(token, key, algorithms, audience=self._audience)` —— 签名 / expiry / audience 三件事由 PyJWT 强制
  3. `sub` claim 必须是 `spiffe://<trust_domain>/...` 格式 + trust_domain 严格匹配
  4. 完整 SPIFFE ID allow-list check
- **JWKS 在 `__init__` 时拉一次** —— SPIRE key rotation 周期 default 24h，启动时 fetch 足够；hot-reload 推后当 "plugin register fails right after rotation" 类 incident 报了再加

**`plugins/registry/config.py` 改动**：
- `K8sSAConfig` / `SpiffeConfig` docstring：从 "Placeholder ... NOT implemented" → 正式 production config
- `build_auth_validator`：删两个 `NotImplementedError` 分支，替换为：
  - `k8s_sa` → `_build_k8s_sa(config.k8s_sa)`：`load_incluster_config`（失败 fallback `load_kube_config`）+ 构造 `AuthenticationV1Api` + 实例化 `K8sSATokenAuth`
  - `spiffe_jwt` → `_build_spiffe(config.spiffe)`：直接传 `jwks_endpoint` / `audience` / `trust_domain` / `trusted_spiffe_ids`
  - 如果 `trusted_sources` 列了某 source 但对应 sub-tree 是 `None` → `ValueError`（把配错捕获在 startup，而不是 runtime validate 时才炸）

**import 策略**：`kubernetes` / `jwt` lib 都**延后到构造时**才 import（`K8sSATokenAuth._call_token_review` 里 import；`_build_k8s_sa` 里 import k8s config/client），所以没启用这两条 source 的环境不会因为缺 K8s lib 加载失败。

**测试**：
- `test_k8s_sa_token.py`（11 tests）：happy path / 空 token / `authenticated=False` / audience 不回显 / SA 不在 allow-list / 非 SA user / 畸形 username / API 爆炸 / 空 audience ValueError / 空 allow-list warning / 请求 body 含 audience
- `test_spiffe_jwt.py`（12 tests）：用 `cryptography` 生成 RSA keypair + 手工签 JWT 跑完整 PyJWT 路径（避免真跑 SPIRE）——happy path / 空 token / aud 错 / expired / 签名错 / sub 非 spiffe / trust domain 错 / SPIFFE ID 不在 allow-list / JWKS fetch 爆炸 / 空 audience ValueError / 空 trust_domain ValueError / 空 allow-list warning
- `test_config.py`：原先 assert `NotImplementedError` 的两个 test 改 assert `ValueError` + match "AuthConfig.{k8s_sa|spiffe} is None"

**验证**：
- `pytest dynamo/planner/tests/plugins/registry/ -q` → 95 passed
- CI-parity → **690 passed, 1 skipped, 11 deselected**（+23 from 667：11 k8s + 12 spiffe）
- proto stubs 无漂移

**剩余 TODO（不阻塞 PR 3.5 ship）**：
- ~~K8s RBAC yaml（`tokenreviews.authentication.k8s.io: ["create"]` verb）—— deployment chart 范围，不在 Python 代码里~~ ✅ 2026-04-28：`deploy/helm/charts/platform/components/operator/templates/planner.yaml` 加 `planner.auth.k8sSA` 开关，`enabled=true` 时 emit ClusterRoleBinding → `system:auth-delegator`（namespaceRestriction 模式 single SA in `.Release.Namespace`；cluster-wide 模式按 `planner.auth.k8sSA.namespaces` 列表，空时 `helm template` 直接 fail-fast）。绑 built-in role 而不是自建 `tokenreviews/create` rule，省维护 + K8s 升级跟版自动同步。registry/README.md `### k8s_sa — TokenReview RBAC` 节带 values.yaml 示例。
- 集成测试：真实 SPIRE or kind cluster —— 卡在 runtime infra（同 PR 7 7-9），先用 mock 单测 unblock ship
- JWKS hot-reload —— 延后到 incident-driven

### v2.0（2026-04-22）—— PR 3 v1 最小集实施完成

**落地范围（9 个 sub-task）**：3-1 types+errors / 3-3 StaticSecretAuth+Multi+AllowUnauth / 3-7 CircuitBreaker / 3-2 RegistryServer (4 RPC + register_internal) / 3-6 HeartbeatMonitor / 3-8 PluginScheduler (含 6 行缓存失效表) / 3-9 ListPlugins 观测字段 / 3-10 config schema + factory / 3-11 integration + README。

**推迟（PR 3.5 follow-up）**：3-4 K8sSATokenAuth / 3-5 SpiffeJwtAuth。`AuthConfig.k8s_sa` / `AuthConfig.spiffe` 的 Pydantic schema **已在 v1 定义**，`build_auth_validator` 收到这两个 source 时抛 `NotImplementedError`（而不是破坏 schema）。

**关键实施决策**：
- **Scheduler ↔ Registry coupling**：原 doc 未指定两者间耦合方式。实施采用 **bilateral subscription**：
  - `Registry.on_unregister(scheduler._on_registry_unregister)` — 缓存失效表行 1/2/4
  - `CircuitBreaker.on_open(scheduler._on_circuit_open)` — 缓存失效表行 3
  - `Registry.attach_cache_age_lookup(scheduler.cache_age)` — 3-9 `PluginInfo.cache_age_seconds`
  - Scheduler 的 `__init__` 自己完成所有 subscription，orchestrator 不需显式 wire
- **`compute_active_set` 触发判定**：原 doc 伪代码 `next_call_due = registered_at + interval * floor((now - registered_at) / interval)` 与 "t=5 不 triggered / t=10 triggered" worked example 不自洽。实施改用简单的 `last_call_at + interval <= now`（初次 `last_call_at=-inf` 总 triggered），与 worked example 一致
- **HOLD_LAST cache 键**：`(plugin_id, stage)` tuple — plugin_type 固定时 stage 只有一个，但 tuple 键让未来多 stage 缓存显式、避免耦合 bug
- **OPEN 事件 fan-out 只在 CLOSED→OPEN 或 HALF_OPEN→OPEN 时触发**（不在已 OPEN 再失败时重复 fan-out）
- **v1 没实现 admin RBAC（Q5 决议：AllowAllAdminAuth 默认）**。`AdminAuthConfig` schema 已定义（mode 字段），build 阶段不处理；PR 5 gRPC gateway / orchestrator 负责调用方鉴权

**测试覆盖**（86 个新测试）：
- 11 auth（static_secret 5 + multi 4 + allow_unauth 2）
- 11 circuit_breaker（全部状态机 transitions + on_open fan-out + 独立性 + config validation）
- 18 server（happy path + heartbeat + unregister + 6 种拒收场景 + register_internal 3 个 case + list_plugins filter + CircuitBreaker reset）
- 9 heartbeat_monitor（uds 剔除 + regular heartbeats + deadline 边界 + late heartbeat reset + builtin in_process skip + **user in_process skip（G-3 v11 回归）** + eviction 走 unregister + run loop round-trip + config validation）
- 14 scheduler：8 active_set + **6 must-pass cache invalidation 行**（行 1 unregister / 行 2 heartbeat 剔除 / 行 3 circuit OPEN / 行 4 version upgrade / 行 5 config reload / 行 6 restart 等价 fresh scheduler）
- 6 list_plugins（cache_age + circuit_state + last_call_at + evaluations_total + transport label + is_builtin）
- 12 config（trusted_sources 空 reject / static_secret / allow_unauth / k8s_sa 未实现 raises / SPIFFE 未实现 raises / build_registry_from_config / protocol_versions / InProcessPluginSpec 的 M-5 extra=forbid + class alias + defaults）
- 5 integration（完整 lifecycle / circuit OPEN→HALF_OPEN 恢复 / heartbeat 剔除 / version upgrade / user in_process 存活 G-3 回归）

**验证**：
- `pytest dynamo/planner/tests/plugins -q` → **272 passed**（186 baseline + 86 PR 3 new）
- CI-parity `pre_merge and planner and gpu_0` → **470 passed, 1 skipped, 11 deselected**
- proto stubs 无漂移
- `components/src/dynamo/planner/plugins/registry/README.md` 含架构图 / 6 行缓存失效表 / auth 决策树 / 单线程 asyncio 不变量 / K8s Secret 部署示例

### v1.2（2026-04-20）—— 主文档 v11 review 同步
- **C-2 v11**：`in_process_plugins` 配置字段从 `SchedulingConfig` 移到 `PluginRegistrationConfig`（与主文档 v11 决议同步）
- **G-3 v11**：HeartbeatMonitor 跳过逻辑改为基于 `transport_type == "in_process"`（不是 `is_builtin`）；`RegisteredPlugin` 数据类加 `transport_type` 字段；3-6 单测 case 加「in_process user plugin 永不剔除」
- **M-5 v11**：`InProcessPluginSpec` Pydantic class 加 `model_config = {"extra": "forbid"}` 拒绝未知字段（包括 `protocol_version`），错误配置快速暴露
- **M-6 v11**：`BuiltinPluginToggle.enabled` 加注释「实际默认值 = enable_*_scaling toggle 的当前值」
- **M-8 v11**：`EndpointsConfig.grpc_listen_addr` 加注释「默认 disable，防意外暴露」

### v1.1（2026-04-20）—— Review 修订（方案 C：P0 + P1-2）
- **P0-2**：3-8 `PluginScheduler` 接口加 single-threaded asyncio 调用约定（class docstring + 每个 method docstring）；明确 `record_result` 必须 `asyncio.gather` 完成后**串行**调，禁止 plugin coroutine 内调
- **P0-3**：3-2 `register()` duplicate plugin_id 决议 reject（不 upsert）；6 行 cache invalidation 表第 4 行重写为 client-driven version upgrade（`unregister` + `register`）；附 Q6 详细决议；附 Q7 CONSTRAIN SET 静态拒收 → runtime drop 决议（与主文档 v9 line 1142-1148 偏差需 cross-check follow-up）
- 其他 review 残留（P1-3 / P1-4 / P1-5 / P2-4）记入 Implementation Breakdown 「PR 1-4 Review 残留问题」节，实施时 mitigation

### v1.0（2026-04-20）—— 初稿
- 与主文档 v10 对齐
- 11 个 sub-task：registry server / 3 种 auth source / heartbeat 监控 / circuit breaker / scheduler / cache invalidation 6 行表 / ListPlugins / 配置
- **Auth 实现按风险分级**：v1 必做 `static_secret`；K8s SA / SPIFFE 标 follow-up 但 schema 必须一并确定（避免 v2 改 schema 兼容代价）
- **Cache invalidation 6 行表 → 6 个独立单测**：每个失效路径必须独立可验证
- **关键不变量**：所有 cache / registry 状态在 single-threaded asyncio event loop 内**无锁**——v9 主文档明确

---

## 为什么 PR 3 风险中等

| 维度 | 评级 | 理由 |
|---|---|---|
| 与现有代码冲突 | **极低** | 全部新建文件；不动 `core/` |
| Auth 接 K8s API（TokenReview） | **中-高** | 需要 RBAC + service account 配置；本地测试需 mock K8s API |
| Cache invalidation 状态空间 | **中** | 6 类失效条件 × 多种 plugin 类型 = 需要细致单测覆盖 |
| Scheduler 时序正确性 | **中** | active set 计算依赖 wall clock；测试要用 VirtualClock 推进 |
| 阻塞下游 | **高** | PR 5 完全依赖；无 PR 3，orchestrator 没有 plugin 来源 |

**核心风险 1**：Auth 与 K8s 集成复杂度——v1 缩减到 `static_secret` 必做，K8s SA / SPIFFE 单独排子任务（可分 PR ship）。

**核心风险 2**：HOLD_LAST cache 6 行失效条件覆盖不全 → 用户 plugin 行为难解释。**唯一缓解**：3-8 子任务 6 行表对应 6 个 must-pass 单测。

---

## 范围

**新建**：
- `components/src/dynamo/planner/plugins/registry/` 目录（server + auth + circuit breaker）
- `components/src/dynamo/planner/plugins/scheduler.py`（active set 计算）
- 配置 schema（auth + scheduling）
- 单测 + 集成测试

**不动**：
- 不动 `core/` / `monitoring/` 任何文件
- 不实现 LocalPlannerOrchestrator（PR 5）
- 不实现真实 builtin plugin（PR 6）—— PR 3 测试用 stub plugin

---

## 子任务清单（11 项）

### 3-1：registry 目录结构 + 数据类型

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/__init__.py`<br/>`components/src/dynamo/planner/plugins/registry/types.py`<br/>`components/src/dynamo/planner/plugins/registry/errors.py` |
| 关键数据类型 | <pre>@dataclass<br/>class RegisteredPlugin:<br/>    """Registry 内部表示，比 PluginInfo 多 runtime 字段"""<br/>    plugin_id: str<br/>    plugin_type: str  # predict/propose/reconcile/constrain<br/>    priority: int<br/>    endpoint: str<br/>    version: str<br/>    protocol_version: str<br/>    execution_interval_seconds: float<br/>    hold_policy: HoldPolicy<br/>    needs: list[str]<br/>    fpm_encoding: str<br/>    request_timeout_seconds: float<br/>    is_builtin: bool<br/>    transport: PluginTransport       # PR 2<br/>    transport_type: Literal["in_process", "uds", "grpc"]  # G-3 v11: HeartbeatMonitor 用此字段决定是否跳过<br/>    registered_at: float       # monotonic<br/>    last_heartbeat_at: float   # monotonic; -inf if never<br/>    last_call_at: float        # monotonic; -inf if never<br/>    evaluations_total: int<br/>    enabled: bool<br/>    # circuit breaker state owned by CircuitBreaker (3-7), referenced via plugin_id<br/><br/>class RegistryError(Exception): pass<br/>class AuthError(RegistryError): pass<br/>class ProtocolVersionError(RegistryError): pass<br/>class DuplicatePluginIdError(RegistryError): pass</pre> |
| 测试 | 无（纯定义） |
| 依赖 | PR 1, PR 2 |
| 估算 | 0.25 天 |

---

### 3-2：PluginRegistryServer 实现

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/server.py` |
| 接口 | <pre>class PluginRegistryServer:<br/>    """Hosts the gRPC service. May also be invoked in-process<br/>       (orchestrator calls register() directly for builtin plugins)."""<br/><br/>    def __init__(<br/>        self,<br/>        clock: Clock,<br/>        auth: AuthValidator,<br/>        circuit_breaker: CircuitBreaker,<br/>        transport_factory: Callable,<br/>        protocol_versions: tuple[str, str] = ("1.0", "1.0"),<br/>    ): ...<br/><br/>    # ─── Public API (called by both gRPC handler and orchestrator) ───<br/>    async def register(self, req: RegisterRequest) -> RegisterResponse: ...<br/>    async def heartbeat(self, plugin_id: str) -> bool: ...<br/>    async def unregister(self, plugin_id: str, reason: str) -> bool: ...<br/>    def list_plugins(self, filter: ListPluginsRequest) -> list[PluginInfo]: ...<br/><br/>    # ─── Internal accessors (orchestrator/scheduler use) ───<br/>    def get_plugin(self, plugin_id: str) -> RegisteredPlugin \| None: ...<br/>    def all_plugins(self) -> list[RegisteredPlugin]: ...<br/><br/>    # ─── Internal register path（builtin / in_process）───<br/>    def register_internal(<br/>        self, plugin_id, plugin_type, priority, instance,<br/>        execution_interval_seconds, hold_policy, **kwargs<br/>    ) -> None:<br/>        """Skips auth + protocol_version checks; wraps instance into<br/>           InProcessTransport; sets is_builtin=True (or per kwargs)."""</pre> |
| `register()` 实施细节 | <ol><li>Auth：调 `auth.validate(req.auth_token)` —— 失败抛 `AuthError`，返回 `RegisterResponse(accepted=false, reject_reason="auth_failed")`</li><li>Protocol version：检查 `req.protocol_version` 在 `protocol_versions` 范围内 —— 否则返回 `RegisterResponse(accepted=false, reject_reason="protocol_version_unsupported: requested=X, supported=[A,B]")`</li><li>**Duplicate check**：`req.plugin_id` 已存在 → 返回 `RegisterResponse(accepted=false, reject_reason="duplicate_plugin_id: client must Unregister before re-Register")`（**P0-3 review 决议：reject，不 upsert**——见下方决议说明）</li><li>**Plugin type guard**：CONSTRAIN plugin 静态拒收 SET——主文档 v9 line 1142-1148 承诺「静态拒」实际不可行（proto 没有 plugin 自报"我会输出 SET"的字段）；v1 实现仅 **runtime drop + audit**（在 type-aware merge `set_allowed=False` 路径，PR 4 4-2 的 `set_dropped` 列表）。本文档与主文档 line 1142-1148 偏差需要 cross-check sub-task 同步主文档</li><li>用 `transport_factory(req.endpoint, req.fpm_encoding)` 创建 transport instance（来自 PR 2 的 `make_transport_for_endpoint`，in_process 路径见 P1-5 review）</li><li>构造 `RegisteredPlugin`，加入内部 dict</li><li>Emit audit log + Prometheus `plugin_register_total{result="accepted"}`</li><li>返回 `RegisterResponse(accepted=true, negotiated_protocol_version=req.protocol_version)`</li></ol> |
| `heartbeat()` 实施细节 | <ol><li>查 plugin_id；不存在返回 `False`</li><li>更新 `last_heartbeat_at = clock.monotonic()`</li><li>返回 `True`</li><li>**注意**：连续缺失检测**不在 heartbeat 调用路径**；交给 3-6 后台 task</li></ol> |
| `unregister()` 实施细节 | <ol><li>查 plugin_id；不存在返回 `False`（idempotent）</li><li>调 `transport.close()`</li><li>从 dict 移除</li><li>**触发 cache invalidation**：通知 `PluginScheduler` 立即清除该 plugin 的 HOLD_LAST cache 与 inherited results（cache 失效 6 行表第 1 行）</li><li>Emit audit log + metric</li></ol> |
| 单测 | `tests/plugins/registry/test_server.py`：<br/>- happy path：register → list → heartbeat → unregister<br/>- duplicate plugin_id 拒收<br/>- protocol_version 超出范围拒收<br/>- auth 失败拒收<br/>- unregister idempotent<br/>- `register_internal` 跳过 auth 但仍走 transport 创建 |
| 依赖 | 3-1, 3-3（auth interface） |
| 估算 | 1.5 天 |

---

### 3-3：Auth source ABC + StaticSecretAuth（v1 必做）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/auth/__init__.py`<br/>`components/src/dynamo/planner/plugins/registry/auth/base.py`<br/>`components/src/dynamo/planner/plugins/registry/auth/static_secret.py`<br/>`components/src/dynamo/planner/plugins/registry/auth/multi.py`（多 source 组合） |
| 接口 | <pre>class AuthValidator(abc.ABC):<br/>    @abc.abstractmethod<br/>    async def validate(self, token: str) -> AuthIdentity: ...<br/>    """Returns identity if valid, raises AuthError otherwise."""<br/><br/>@dataclass<br/>class AuthIdentity:<br/>    source: str       # "static_secret" / "k8s_sa" / "spiffe_jwt"<br/>    subject: str      # service account or SPIFFE ID or "static"<br/>    metadata: dict<br/><br/>class StaticSecretAuth(AuthValidator):<br/>    def __init__(self, secrets: dict[str, str]):<br/>        """secrets: { secret_value: subject_label }"""<br/>        self._secrets = secrets<br/><br/>    async def validate(self, token: str) -> AuthIdentity:<br/>        if token in self._secrets:<br/>            return AuthIdentity("static_secret", self._secrets[token], {})<br/>        raise AuthError("static_secret: token not in trusted set")<br/><br/>class MultiSourceAuth(AuthValidator):<br/>    def __init__(self, sources: list[AuthValidator]):<br/>        self._sources = sources<br/><br/>    async def validate(self, token: str) -> AuthIdentity:<br/>        last_err = None<br/>        for src in self._sources:<br/>            try: return await src.validate(token)<br/>            except AuthError as e: last_err = e<br/>        raise AuthError(f"all sources failed; last: {last_err}")<br/><br/>class AllowUnauthenticatedAuth(AuthValidator):<br/>    """Dev-only fallback. Logs WARNING on init."""<br/>    async def validate(self, token: str) -> AuthIdentity:<br/>        return AuthIdentity("allow_unauthenticated", "anonymous", {})</pre> |
| **关键设计决策** | <ol><li>**`MultiSourceAuth` 短路**：第一个成功的 source 即返回；用户配 `[k8s_sa, static_secret]` 后 K8s 通过即不查 static</li><li>**`AllowUnauthenticatedAuth` 启动时 WARNING log**：避免运维不小心带去 prod；建议 startup script 在生产环境 grep log 阻止启动</li><li>**Token 中不暴露具体 subject**：拒绝时 `reject_reason` 仅含 "auth_failed"，不写明哪一步失败（防止 token 泄漏给攻击者作 oracle）</li></ol> |
| 单测 | `tests/plugins/registry/auth/test_static_secret.py`：<br/>- 已知 secret 通过<br/>- 未知 secret 拒收<br/>- 空 token 拒收<br/>`tests/plugins/registry/auth/test_multi.py`：<br/>- 第一个 source 成功 → 后续不调<br/>- 所有失败 → 抛 AuthError，含最后一个错误<br/>`tests/plugins/registry/auth/test_allow_unauth.py`：<br/>- WARNING log 在 init 时写出 |
| 依赖 | 3-1 |
| 估算 | 0.5 天 |

---

### 3-4：K8sSATokenAuth（follow-up 可独立 ship）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/auth/k8s_sa.py` |
| 接口 | <pre>class K8sSATokenAuth(AuthValidator):<br/>    def __init__(<br/>        self,<br/>        kube_client,  # kubernetes.client.AuthenticationV1Api<br/>        audience: str,<br/>        trusted_service_accounts: set[str],  # "ns/sa"<br/>    ): ...<br/><br/>    async def validate(self, token: str) -> AuthIdentity:<br/>        # 1. Call TokenReview API<br/>        review = await self._call_token_review(token, self._audience)<br/>        if not review.status.authenticated:<br/>            raise AuthError("k8s_sa: TokenReview rejected")<br/>        # 2. Verify SA in allow-list<br/>        sa = self._extract_sa(review)  # "namespace/serviceaccount"<br/>        if sa not in self._trusted_service_accounts:<br/>            raise AuthError(f"k8s_sa: SA {sa} not in allow-list")<br/>        return AuthIdentity("k8s_sa", sa, {"audience": self._audience})</pre> |
| **关键设计决策** | <ol><li>**TokenReview API 必须 cluster-scope** —— planner Pod 需要 `system:auth-delegator` ClusterRole；3-4 子任务**附 RBAC yaml 模板**到 `deploy/helm/` 里</li><li>**audience 校验** —— prevent token confusion attack：projected SA token 必须显式 audience；planner 配置时审计</li><li>**TokenReview 缓存**：hot path 每次都调 K8s API 太重；单 token 缓存 5 分钟（**v1 暂不实现**，标 follow-up）</li><li>**异常处理**：K8s API 不可达 / 超时 → `AuthError("k8s_sa: API unreachable")` —— 不 fall through 到其他 source（防止 K8s 故障被静默放行）</li></ol> |
| 单测 | `tests/plugins/registry/auth/test_k8s_sa.py`：<br/>- mock TokenReview API：authenticated + SA 在 list → 通过<br/>- authenticated 但 SA 不在 list → 拒收<br/>- not authenticated → 拒收<br/>- audience 不匹配 → 拒收<br/>- API timeout → 拒收<br/>集成测试（kind cluster fixture，**可选**）：跑实际 K8s API |
| 依赖 | 3-3 |
| 估算 | 1.5 天 |
| **可推迟** | ✅ —— 如果 v1 ship 卡时间，K8s SA 标为 follow-up PR 3.5；v1 仅 static_secret + dev 用 |

---

### 3-5：SpiffeJwtAuth（follow-up 可独立 ship）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/auth/spiffe.py` |
| 接口 | <pre>class SpiffeJwtAuth(AuthValidator):<br/>    def __init__(<br/>        self,<br/>        jwks_endpoint: str,    # SPIRE JWKS<br/>        audience: str,<br/>        trusted_spiffe_ids: set[str],  # "spiffe://example.com/planner-plugin"<br/>    ): ...<br/><br/>    async def validate(self, token: str) -> AuthIdentity:<br/>        # 1. Verify JWT signature against JWKS<br/>        try: claims = jwt.decode(token, jwks=self._jwks, audience=self._audience)<br/>        except Exception as e: raise AuthError(...) from e<br/>        # 2. Verify SPIFFE ID in allow-list<br/>        sid = claims["sub"]  # spiffe://...<br/>        if sid not in self._trusted_spiffe_ids:<br/>            raise AuthError(...)<br/>        return AuthIdentity("spiffe_jwt", sid, {"audience": self._audience})</pre> |
| **关键设计决策** | <ol><li>**依赖 `pyjwt` 库**——dynamo 是否已引入需先调研；否则 PR 3 加新依赖（可接受）</li><li>**JWKS 缓存**：v1 启动时从 endpoint 拉一次；**不**实现 hot reload（SPIRE keys 一般 rotation 周期长）；v2 follow-up 加 inotify or periodic refetch</li><li>**Trust domain 校验**：`spiffe://example.com/...` 中 `example.com` 必须匹配配置；3-5 子任务**额外加 trust_domain 配置项**</li></ol> |
| 单测 | `tests/plugins/registry/auth/test_spiffe.py`：<br/>- valid JWT + SPIFFE ID 在 list → 通过<br/>- 无效签名 → 拒收<br/>- audience 不匹配 → 拒收<br/>- SPIFFE ID 不在 list → 拒收 |
| 依赖 | 3-3 |
| 估算 | 1 天 |
| **可推迟** | ✅ —— v1 ship 可不含；follow-up |

---

### 3-6：Heartbeat 监控后台 task

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/heartbeat_monitor.py` |
| 接口 | <pre>class HeartbeatMonitor:<br/>    def __init__(<br/>        self,<br/>        registry: PluginRegistryServer,<br/>        scheduler: PluginScheduler,<br/>        clock: Clock,<br/>        timeout_seconds: float = 15.0,<br/>        missed_threshold: int = 2,<br/>        check_interval_seconds: float = 5.0,<br/>    ): ...<br/><br/>    async def run(self):<br/>        """Long-running coroutine; orchestrator schedules at startup."""<br/>        while not self._stopped:<br/>            await self._check_once()<br/>            await self._clock.sleep(self._check_interval_seconds)<br/><br/>    async def _check_once(self):<br/>        now = self._clock.monotonic()<br/>        for p in self._registry.all_plugins():<br/>            # G-3 review (v11): 跳过逻辑基于 transport，不是 is_builtin<br/>            # 否则 in_process **user** plugin (is_builtin=False) 因不发<br/>            # heartbeat 会被立即剔除。<br/>            if p.transport_type == "in_process":<br/>                continue<br/>            elapsed = now - p.last_heartbeat_at<br/>            if elapsed > self._timeout_seconds * self._missed_threshold:<br/>                # Evict<br/>                await self._registry.unregister(p.plugin_id, reason="heartbeat_missed")<br/>                # registry.unregister already triggers scheduler cache clear (3-2)</pre> |
| **关键设计决策** | <ol><li>**`transport_type == "in_process"` 跳过 heartbeat 检查**（G-3 review v11）——in-process 不存在 "丢失" 问题；**包含 builtin 与 in_process user plugin 两类**（之前误用 `is_builtin` 会让 in_process user plugin 因不发心跳被立即剔除）</li><li>**`missed_threshold=2`**：连续 2 个 timeout 窗口（默认 30s）才剔除；防止单次抖动剔人</li><li>**用 `Clock.sleep`**——VirtualClock 测试中可推进时间快速验证 missed eviction</li><li>**剔除走标准 unregister 路径**——保证 cache 清理 + audit log 一致</li></ol> |
| 单测 | `tests/plugins/registry/test_heartbeat_monitor.py`（用 VirtualClock）：<br/>- 初始注册（uds transport）→ 无 heartbeat → advance 30s → plugin 被剔除<br/>- 每 5s 一次 heartbeat → advance 60s → 不剔除<br/>- 连续 1 次 miss + 1 次 heartbeat → 不剔除（reset 计数）<br/>- builtin plugin（transport_type=in_process）→ 永不剔除<br/>- **in_process user plugin（is_builtin=False, transport_type=in_process）→ 永不剔除**（G-3 review v11 验证）<br/>- 剔除路径触发 scheduler cache clear（mock scheduler 验证调用）|
| 依赖 | 3-2 |
| 估算 | 1 天 |

---

### 3-7：CircuitBreaker（3 状态机）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/circuit_breaker.py` |
| 接口 | <pre>class CircuitBreaker:<br/>    """Per-plugin circuit breaker.<br/>       CLOSED -> [N consecutive failures] -> OPEN<br/>       OPEN -> [cooldown elapsed] -> HALF_OPEN<br/>       HALF_OPEN -> [1 success] -> CLOSED<br/>       HALF_OPEN -> [1 failure] -> OPEN (reset cooldown)"""<br/><br/>    def __init__(<br/>        self,<br/>        clock: Clock,<br/>        failure_threshold: int = 5,<br/>        cooldown_seconds: float = 30.0,<br/>    ): ...<br/><br/>    def state(self, plugin_id: str) -> CircuitState: ...<br/>    def can_call(self, plugin_id: str) -> bool: ...<br/>    def record_success(self, plugin_id: str) -> None: ...<br/>    def record_failure(self, plugin_id: str) -> None: ...<br/>    def reset(self, plugin_id: str) -> None: ...  # used on register/unregister</pre> |
| **关键设计决策** | <ol><li>**failure_threshold = 5 / cooldown = 30s**：默认值参考主文档；可配</li><li>**HALF_OPEN 只允许 1 次试探**——多次试探放给后续 PR 5 调度器决定（**v1 简化版**）</li><li>**用 `Clock.monotonic()`** 计算 cooldown elapsed（不受 NTP 影响）</li><li>**state 存内存**——v9 cache 持久化表第 3 行明确"重启全清，回到 CLOSED"</li><li>**reset 在 unregister 时调**——防止 plugin_id 重新注册时继承旧 OPEN 状态</li></ol> |
| 单测 | `tests/plugins/registry/test_circuit_breaker.py`（用 VirtualClock）：<br/>- 5 次连续失败 → OPEN<br/>- OPEN 状态 `can_call=False`<br/>- advance(30s) → HALF_OPEN<br/>- HALF_OPEN 1 次成功 → CLOSED<br/>- HALF_OPEN 1 次失败 → OPEN，cooldown 重置<br/>- reset 在任何状态下回到 CLOSED |
| 依赖 | PR 2 |
| 估算 | 0.75 天 |

---

### 3-8：PluginScheduler（active set + cache invalidation 6 行表）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/scheduler.py` |
| 接口 | <pre>class PluginScheduler:<br/>    """Single-threaded asyncio invariant (P0-2 review 决议):<br/>      ALL methods MUST be called from the event loop main task.<br/>      Concurrent record_result / invalidate_cache from multiple<br/>      asyncio tasks (e.g. inside `asyncio.gather` plugin coroutines)<br/>      is UNDEFINED BEHAVIOR. Orchestrator MUST serialize:<br/>        results = await asyncio.gather(*plugin_calls)<br/>        for r in results:  # serial here<br/>            scheduler.record_result(...)<br/>      No locks needed BY ASSUMPTION; tests in 5-9 verify (assert<br/>      asyncio.current_task() is the main task)."""<br/><br/>    def __init__(<br/>        self,<br/>        registry: PluginRegistryServer,<br/>        circuit_breaker: CircuitBreaker,<br/>        clock: Clock,<br/>    ): ...<br/><br/>    def compute_active_set(<br/>        self, now: float, stage: str<br/>    ) -> ActiveSet:<br/>        """Returns (triggered_plugins, inherited_results) for this stage<br/>           at this tick. Sync; must be called from event loop main task."""<br/>        ...<br/><br/>    def record_result(<br/>        self, plugin_id: str, stage: str, result: OverrideResult, tick_now: float<br/>    ) -> None:<br/>        """Called after pipeline runs the plugin; updates HOLD_LAST cache.<br/>           Sync; must be called from event loop main task AFTER<br/>           asyncio.gather completes (NOT inside plugin coroutine)."""<br/>        ...<br/><br/>    def invalidate_cache(self, plugin_id: str, reason: str) -> None:<br/>        """Called by registry events / config reload.<br/>           Sync; must be called from event loop main task."""<br/>        ...<br/><br/>@dataclass<br/>class ActiveSet:<br/>    triggered: list[RegisteredPlugin]   # 本 tick 调用的 plugin<br/>    inherited: list[InheritedResult]    # 上次 cache 注入的结果<br/><br/>@dataclass<br/>class InheritedResult:<br/>    plugin_id: str<br/>    priority: int<br/>    result: OverrideResult<br/>    cached_at: float</pre> |
| `compute_active_set` 算法 | <ol><li>遍历 registry 中**所有** stage 匹配的 plugin（含 `enabled=False` 过滤）</li><li>对每个 plugin：<ul><li>检查 `circuit_breaker.can_call(plugin_id)`：False → 跳过（既不 triggered 也不 inherited）</li><li>计算 `next_call_due`：`registered_at + execution_interval_seconds * floor((now - registered_at) / execution_interval_seconds)`</li><li>如果 `next_call_due <= now`：加入 `triggered`</li><li>否则：检查 hold_policy<ul><li>`HOLD_LAST` 且 cache 中有有效结果：加入 `inherited`</li><li>`ACCEPT_WHEN_IDLE` 或 cache 空：跳过（视为 ACCEPT）</li></ul></li></ul></li></ol> |
| **Cache invalidation 6 行表** —— 必须每个独立单测 | <table><tr><th>触发条件</th><th>清除范围</th></tr><tr><td>1. plugin Unregister</td><td>该 plugin 全部 cache</td></tr><tr><td>2. heartbeat missed → 自动剔除（走 unregister 路径）</td><td>同上</td></tr><tr><td>3. circuit breaker → OPEN</td><td>该 plugin 全部 cache</td></tr><tr><td>4. plugin 主动 Unregister + 紧接 Register（client-driven version upgrade；**P0-3 决议**：v1 不支持 server-side upsert，client 必须先 Unregister）</td><td>cache 在 unregister 时自然清；新一轮 register 重新积累</td></tr><tr><td>5. 配置 reload (`config.reload()`)</td><td>**全部** plugin cache（保险起见）</td></tr><tr><td>6. orchestrator restart</td><td>cache 内存全清（main 进程退出）</td></tr></table> |
| 单测 | `tests/plugins/scheduler/test_active_set.py`（用 VirtualClock）：<br/>- 单 plugin `execution_interval=10s`：t=0 register → t=5 不 triggered → t=10 triggered → t=15 inherited (HOLD_LAST) / 跳过 (ACCEPT_WHEN_IDLE)<br/>- 多 plugin 不同 interval：每 tick active set 正确<br/>- circuit OPEN plugin 不出现在 active set<br/>- enabled=False plugin 不出现在 active set<br/><br/>`tests/plugins/scheduler/test_cache_invalidation.py`：**6 行表逐行验证**<br/>- 行 1：unregister → cache 清<br/>- 行 2：heartbeat missed → unregister → cache 清<br/>- 行 3：record 5 failures → circuit OPEN → cache 清<br/>- 行 4：client-driven version upgrade —— `unregister(plugin_id, reason="version_upgrade")` → cache 清 → `register(plugin_id, version=new)` → 新 plugin 重新积累；server 不做 upsert（**P0-3 决议**）<br/>- 行 5：调 `invalidate_cache(reason="config_reload")` → 全清<br/>- 行 6：integration test 跑完 close orchestrator → 重启 → cache 空 |
| 依赖 | 3-2, 3-7 |
| 估算 | 2 天（含 6 行表 6 个 must-pass 单测）|

---

### 3-9：ListPlugins 实现 + admin RBAC

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/server.py`（list_plugins method）<br/>`components/src/dynamo/planner/plugins/registry/admin_auth.py`（admin RBAC validator） |
| 接口 | <pre>def list_plugins(<br/>    self, filter: ListPluginsRequest<br/>) -> list[PluginInfo]:<br/>    """Convert RegisteredPlugin → PluginInfo (proto), apply filter."""<br/>    out = []<br/>    for p in self._plugins.values():<br/>        if filter.stage_filter and p.plugin_type != filter.stage_filter:<br/>            continue<br/>        if not filter.include_disabled and not p.enabled:<br/>            continue<br/>        out.append(PluginInfo(<br/>            plugin_id=p.plugin_id,<br/>            plugin_type=p.plugin_type,<br/>            priority=p.priority,<br/>            version=p.version,<br/>            protocol_version=p.protocol_version,<br/>            enabled=p.enabled,<br/>            is_builtin=p.is_builtin,<br/>            transport=self._transport_label(p.endpoint),<br/>            circuit_state=self._circuit_breaker.state(p.plugin_id),<br/>            evaluations_total=p.evaluations_total,<br/>            last_call_at_seconds_ago=self._clock.monotonic() - p.last_call_at,<br/>            cache_age_seconds=self._scheduler.cache_age(p.plugin_id),<br/>        ))<br/>    return out</pre> |
| **关键设计决策** | <ol><li>**admin RBAC 与 plugin Register auth 分离**——`ListPlugins` 是**运维 / 调试 endpoint**；权限粒度不同（K8s ClusterRole `dynamo:planner-admin`）</li><li>**v1 admin auth 简化版**：`AllowAllAdminAuth`（默认）+ `K8sRBACAdminAuth`（follow-up）；schema 字段 `planner.plugin_registration.admin.auth` 已预留</li><li>**`last_call_at_seconds_ago` 与 `cache_age_seconds`**：靠 scheduler 提供（3-8 加 `cache_age()` accessor）</li></ol> |
| 单测 | `tests/plugins/registry/test_list_plugins.py`：<br/>- 注册 5 个 plugin（混合 stage + builtin） → list 返回 5<br/>- stage_filter='propose' → 仅 propose 类返回<br/>- include_disabled=False → enabled=False plugin 不返回<br/>- circuit_state / cache_age 字段正确填充 |
| 依赖 | 3-2, 3-7, 3-8 |
| 估算 | 0.5 天 |

---

### 3-10：配置 schema（auth + scheduling）

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/registry/config.py` |
| 接口 | <pre>class PluginRegistrationConfig(BaseModel):<br/>    """planner.plugin_registration 配置树（v11 决议：in_process_plugins 在此节）"""<br/>    endpoints: EndpointsConfig<br/>    auth: AuthConfig<br/>    transport: TransportConfig  # 来自 PR 2-7<br/>    protocol_version_min: str = "1.0"<br/>    protocol_version_max: str = "1.0"<br/>    heartbeat_timeout_seconds: float = 15.0<br/>    heartbeat_missed_threshold: int = 2<br/>    in_process_plugins: list[InProcessPluginSpec] = []  # v11 移到此处（与 user plugin 注册同一概念聚合）<br/>    admin: AdminAuthConfig<br/><br/>class EndpointsConfig(BaseModel):<br/>    uds_socket_path: str = "/var/run/dynamo/planner/registry.sock"<br/>    grpc_listen_addr: str \| None = None  # ":50051" or None to disable（默认 disable，防意外暴露）<br/><br/>class AuthConfig(BaseModel):<br/>    trusted_sources: list[Literal[<br/>        "k8s_sa", "spiffe_jwt", "static_secret", "allow_unauthenticated"<br/>    ]] = []  # default = empty = reject all<br/>    static_secrets: dict[str, str] = {}  # secret -> subject_label<br/>    k8s_sa: K8sSAConfig \| None = None       # v1 不可用；PR 3.5 follow-up<br/>    spiffe: SpiffeConfig \| None = None     # v1 不可用；PR 3.5 follow-up<br/><br/>class K8sSAConfig(BaseModel):<br/>    audience: str = "dynamo-planner"<br/>    trusted_service_accounts: list[str] = []  # "ns/sa" strings<br/><br/>class SpiffeConfig(BaseModel):<br/>    jwks_endpoint: str<br/>    audience: str = "dynamo-planner"<br/>    trust_domain: str  # "example.com"<br/>    trusted_spiffe_ids: list[str] = []<br/><br/>class SchedulingConfig(BaseModel):<br/>    """planner.scheduling 配置树（v11: in_process_plugins 已移到 PluginRegistrationConfig）"""<br/>    clock: ClockConfig  # 来自 PR 2-7<br/>    request_timeout_seconds: float = 5.0<br/>    tick_max_duration_seconds: float = 30.0<br/>    builtins: dict[str, BuiltinPluginToggle] = {}  # plugin_id -> toggle<br/><br/>class InProcessPluginSpec(BaseModel):<br/>    """In-process user plugin 配置 spec（v11 移到 PluginRegistrationConfig）"""<br/>    model_config = {"extra": "forbid"}  # M-5 实施注解：禁止未知字段<br/>    module: str                                  # python module path<br/>    class_: str = Field(..., alias="class")     # plugin class name in module<br/>    plugin_id: str                               # globally unique<br/>    plugin_type: Literal["predict", "propose", "reconcile", "constrain"]<br/>    priority: int<br/>    execution_interval_seconds: float<br/>    hold_policy: Literal["ACCEPT_WHEN_IDLE", "HOLD_LAST"] = "ACCEPT_WHEN_IDLE"<br/>    kwargs: dict[str, Any] = {}<br/>    # M-5 实施注解：不接受 `protocol_version` 字段——in_process plugin 编译时绑定<br/><br/>class BuiltinPluginToggle(BaseModel):<br/>    enabled: bool = True  # M-6 注解：实际默认值 = enable_*_scaling toggle 的当前值<br/>    priority: int \| None = None  # override default<br/>    execution_interval_seconds: float \| None = None</pre> |
| **配置 → registry 工厂** | <pre>def build_registry_from_config(<br/>    config: PluginRegistrationConfig, clock: Clock<br/>) -> PluginRegistryServer:<br/>    auth = build_auth_validator(config.auth)<br/>    cb = CircuitBreaker(clock)<br/>    return PluginRegistryServer(<br/>        clock=clock, auth=auth, circuit_breaker=cb,<br/>        transport_factory=...,<br/>        protocol_versions=(config.protocol_version_min, config.protocol_version_max),<br/>    )<br/><br/>def build_auth_validator(c: AuthConfig) -> AuthValidator:<br/>    if not c.trusted_sources:<br/>        raise ValueError("auth.trusted_sources empty: registry will reject all (insecure config)")<br/>    sources = []<br/>    for src in c.trusted_sources:<br/>        if src == "static_secret":<br/>            sources.append(StaticSecretAuth(c.static_secrets))<br/>        elif src == "k8s_sa":<br/>            sources.append(K8sSATokenAuth(...))<br/>        elif src == "spiffe_jwt":<br/>            sources.append(SpiffeJwtAuth(...))<br/>        elif src == "allow_unauthenticated":<br/>            sources.append(AllowUnauthenticatedAuth())<br/>            log.warning("AllowUnauthenticatedAuth enabled — DEV ONLY")<br/>    return MultiSourceAuth(sources)</pre> |
| 单测 | `tests/plugins/registry/test_config.py`：<br/>- 默认配置（trusted_sources=[]）→ build 抛 ValueError<br/>- `[static_secret]` + 空 secrets → 启动成功但任何 token 都拒（warning log）<br/>- `[allow_unauthenticated]` → warning log<br/>- 多 source 正确组合为 `MultiSourceAuth` |
| 依赖 | 3-2, 3-3, 3-7, 3-8 |
| 估算 | 0.75 天 |

---

### 3-11：集成测试 + README

| 项 | 内容 |
|---|---|
| 集成测试位置 | `tests/integration/test_registry_e2e.py` |
| 集成测试 | <ol><li>启 PluginRegistryServer (uds endpoint)</li><li>启 echo plugin client（用 PR 2 的 UdsTransport 反向）→ Register → Heartbeat × 5 → Unregister</li><li>验证 ListPlugins 在每个阶段返回正确状态</li><li>测试 cache 失效：Register → record_result → Unregister → 验证下一 tick active set 不含该 plugin 也不含 inherited</li><li>测试 circuit breaker：连续 5 次失败 → 第 6 次 active set 不含该 plugin → advance(30s) → HALF_OPEN → 1 次成功 → CLOSED</li></ol> |
| README 位置 | `components/src/dynamo/planner/plugins/registry/README.md` |
| README 内容 | <ol><li>PluginRegistry 架构图（server + 4 RPC + auth + heartbeat + circuit breaker + scheduler 的关系）</li><li>3 种 auth source 选择决策树（dev: static / sidecar: k8s_sa / mesh: spiffe）</li><li>Cache invalidation 6 行表（运维 / debug 必读）</li><li>Plugin 注册的 happy path 与失败案例（auth 失败 / version 不兼容 / duplicate id 等）</li><li>K8s deployment 示例：planner Pod RBAC（如果用 k8s_sa）+ secret mount（如果用 static_secret） |
| 依赖 | 3-2 ~ 3-10 全部完成 |
| 估算 | 1.5 天（集成测试 1 天 + README 0.5 天） |

---

## PR 3 总估算

- **单人**：~12 天（3-1: 0.25 + 3-2: 1.5 + 3-3: 0.5 + 3-4: 1.5 + 3-5: 1 + 3-6: 1 + 3-7: 0.75 + 3-8: 2 + 3-9: 0.5 + 3-10: 0.75 + 3-11: 1.5 ≈ 11.25 天，保守 12 天 ≈ 2.5 周）
- **双人并行**：~7-8 天
  - 工程师 A：3-1 → 3-2 → 3-9 → 3-10 → 3-11（registry / config / 集成）
  - 工程师 B：3-3 → 3-7 → 3-8（auth / circuit / scheduler）
  - 可推迟：3-4 / 3-5 标 follow-up，v1 仅必做（约省 2.5 天）
- **v1 最小集**（推迟 K8s SA + SPIFFE 到 follow-up）：~9.5 天单人 / ~5-6 天双人并行

注：原 Implementation Breakdown 估算 1 工程师 × 1.5-2 周（≈ 7.5-10 天）有些乐观，因为没单独算 cache invalidation 6 行表的测试工作量（约 1 天）。本 PR 3 详细文档建议按 **单人 12 天 / 双人 7-8 天** 排期。

---

## PR 3 Acceptance Criteria

### v1 必做集合
- [ ] `PluginRegistryServer` 4 RPC（Register / Heartbeat / Unregister / ListPlugins）全部实现
- [ ] `StaticSecretAuth` 实现 + 单测
- [ ] `MultiSourceAuth` + `AllowUnauthenticatedAuth` 实现
- [ ] `HeartbeatMonitor` 后台 task，VirtualClock 单测覆盖剔除场景
- [ ] `CircuitBreaker` 3 状态机，VirtualClock 单测覆盖全部 transition
- [ ] `PluginScheduler.compute_active_set` 实现 + active set 单测
- [ ] **Cache invalidation 6 行表 6 个 must-pass 单测全部通过**
- [ ] `ListPlugins` 实现 + filter 单测
- [ ] 配置 schema + `build_registry_from_config` 工厂
- [ ] 集成测试 e2e 通过
- [ ] README 完整覆盖架构图 + 6 行表 + 部署示例
- [ ] CI `planner-test` job 自动包含全部测试

### v1 可推迟（follow-up PR 3.5）
- [ ] `K8sSATokenAuth`（含 RBAC yaml + 集成测试）
- [ ] `SpiffeJwtAuth`（含 trust domain 校验 + 集成测试）
- [ ] `K8sRBACAdminAuth`（admin endpoint）

---

## 跨 Sub-task 必须协调的点

### 1. CircuitBreaker / Scheduler / Registry 三方耦合

cache invalidation 触发链：
```
plugin failure (PR 5 调用层)
  → CircuitBreaker.record_failure (3-7)
  → 检测达到 threshold → 内部状态 OPEN
  → CircuitBreaker emit event → Scheduler.invalidate_cache (3-8)
```

3-7 与 3-8 必须协调 event 通知机制：
- **方案 A（决议）**：CircuitBreaker 持有 callback list；Scheduler 在初始化时 subscribe；OPEN 状态触发时 fan-out
- **方案 B（拒绝）**：Scheduler 主动 poll CircuitBreaker state——增加调用频率，并发难处理

### 2. Heartbeat 监控与 Unregister 共享代码路径

3-6 检测 missed heartbeat → 调 `registry.unregister(plugin_id, reason="heartbeat_missed")`，**走与用户主动 unregister 同一代码路径**——保证 cache 清理 / audit log 一致。任何修改 unregister 行为必须同时验证 heartbeat 触发场景。

### 3. Auth source 加新类型时的契约

未来如果加 `OAuth2Auth` 等新 source：
1. 实现 `AuthValidator` interface
2. 在 `AuthConfig.trusted_sources` Literal 加新 string
3. 在 `build_auth_validator` 加新 case
4. 加单测 + 集成测试 + README 更新

3-3 / 3-4 / 3-5 共享同一 ABC，加新源**不影响**已有源。

### 4. config 默认值 = 安全（fail closed）

`auth.trusted_sources=[]` → 启动抛 ValueError；不允许"忘配置→默认放行"。这是与主文档 v9 line 1209 一致的设计原则，3-10 必须强制。

---

## 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| K8s SA TokenReview 接入复杂（RBAC / audience 配置错） | **中-高** | 3-4 单独排子任务；v1 不必做（static_secret 已足够 dev / 单 cluster prod）；follow-up PR 3.5 处理 |
| Cache invalidation 漏失效路径 → 用户 plugin "鬼魂" 仍被调 | **中** | 3-8 6 行表对应 6 个 must-pass 单测；任何 cache 相关 PR 必须保留这 6 行覆盖 |
| Heartbeat monitor 后台 task 卡死 → 死 plugin 不剔除 | **中** | 3-6 加 health metric `heartbeat_check_last_run_at`；超过 2 × check_interval 触发告警 |
| CircuitBreaker 太激进（5 次失败立即 OPEN）→ 偶发故障被放大 | **低** | 默认 5/30s 偏激进；在 README 说明可调；推荐 production 提高到 10/60s |
| protocol_version 协商错误 → 旧 plugin 重启后被拒 | **低** | 3-2 必须保留 `protocol_versions` 双向支持窗口（≥ 2 个连续版本）；v2 升级时启用 |

---

## 推荐 staffing

- **1 名 Backend 工程师**（K8s + RBAC 经验）：负责 3-2 / 3-4 / 3-9 / 3-10 / 3-11
- **1 名 Backend 工程师**（asyncio + 测试经验）：负责 3-3 / 3-6 / 3-7 / 3-8
- 可推迟 SPIFFE 给 mesh 团队（如果有）：3-5

---

## Resolved Questions（已决议）

### Q1：v1 是否必须含 K8s SA / SPIFFE？

**决议**：**否，v1 仅必做 static_secret**。理由：
- 大部分 dynamo 部署是 single-cluster；static_secret 配合 K8s Secret mount 已经够安全
- K8s SA / SPIFFE 增加运维复杂度（RBAC、audience、SPIRE 部署）；可作 follow-up 渐进引入
- schema 字段（K8sSAConfig / SpiffeConfig）已在 v1 定义，未来加 source 不破坏配置兼容

### Q2：CircuitBreaker 状态是否要持久化？

**决议**：**不**（v1）。理由：
- 主文档 v9 cache 持久化表第 3 行明确"重启全清，回到 CLOSED"
- circuit state 持久化需要外部存储（etcd / file），引入新依赖
- 重启后所有 plugin 第一次调如果还是失败，circuit 会在 5 次后 OPEN——只是延迟 5 秒，可接受
- 如果未来需要"重启快速恢复 OPEN 状态"，加可选 file-based persistence（不影响 v1 接口）

### Q3：HeartbeatMonitor 与 PluginScheduler 之间的调用方向？

**决议**：HeartbeatMonitor → registry.unregister → registry 内部触发 scheduler.invalidate_cache。**不**让 HeartbeatMonitor 直接调 scheduler。理由：
- 单一入口（unregister）保证审计日志一致
- 解耦 monitor / scheduler

### Q4：`PluginScheduler` 是否要支持 plugin 的「按需触发」（除了周期）？

**决议**：**否**（v1）。理由：
- 主文档 v9 仅定义周期触发（`execution_interval_seconds`）；按需触发是 v2 feature
- v1 简化让验证更容易；v2 加 manual trigger API 不破坏 v1 客户端

### Q5：admin endpoint（ListPlugins）是否走 PluginRegistry 同一 RPC server？

**决议**：**v1 同一 server，但 admin auth 独立**。理由：
- 同一 server 简化部署（一个 grpc port）
- admin auth 独立避免 plugin auth 泄漏 admin 权限
- 未来可拆 admin 到独立 server（不影响协议）

### Q6：duplicate plugin_id —— reject vs server-side upsert？（P0-3 review 决议）

**决议**：**v1 reject duplicate**（`reject_reason="duplicate_plugin_id"`）。client 必须按「Unregister 旧 → Register 新」流程升级 plugin 版本。

理由：
- **upsert 原子性难保证**——如果新 register 中途失败（auth / version / endpoint 不通），旧 plugin 已 unregister；要 rollback 旧的复杂度高且容易死锁
- **客户端 explicit 比 server implicit 安全**——client 知道何时升级，可控制流量切换时机；server upsert 可能在 plugin 恰好被调用时切换
- **K8s rolling update 友好**——新 Pod 启动 → readiness probe 通过 → 旧 Pod 收 SIGTERM → 优雅 Unregister → 新 Pod Register；**自然就是「先 Unregister 再 Register」**
- **client 实现成本低**——SDK 层封装一个 `update_version(new_version)` helper 即可（`unregister + register`，错误时 retry register）

**与 cache invalidation 6 行表第 4 行一致**：行 4 描述「client-driven version upgrade」走 unregister 路径自然清 cache；server 不引入隐式 upsert 路径。

### Q7：CONSTRAIN plugin SET 拒收 —— 静态 vs runtime？（P0-3 review 副带决议）

**决议**：**v1 仅 runtime drop + audit**（在 type-aware merge `set_allowed=False` 路径处理；PR 4 4-2 实现）。

理由：
- 主文档 v9 line 1142-1148 承诺「register 阶段静态拒」**不可行**：proto3 没有「plugin 自报会输出哪些 OverrideType」字段；server 无法在 register 时知道 plugin 行为
- runtime drop 已经够用——audit log + Prometheus `plugin_constrain_set_dropped_total` 让运维追踪滥用 plugin
- **cross-check 同步**：主文档 line 1142-1148 需在下次主文档修订时改为「runtime drop + audit」（不在本 PR 3 范围；记入 cross-check report follow-up）

---

## 已删除的内容（v0 → v1）

无（首版）

---

## 与其他 PR 的接口

| 下游 PR | 依赖 PR 3 的内容 |
|---|---|
| PR 5（Orchestrator） | 整个 `PluginRegistryServer` + `PluginScheduler` + `CircuitBreaker`；orchestrator 在 init 时构造 registry，每 tick 调 `compute_active_set`，调用结果用 `record_result` 回写 cache |
| PR 6（5 个 builtin） | `register_internal` 路径注册 builtin；`PluginLifecycle.Bootstrap` 在 register 后由 orchestrator 调（PR 5 实现） |
| PR 7（NativePlannerBase 双路径） | 启动代码用 `build_registry_from_config` 构造 registry；config 来源仍是现有 `PlannerConfig` 扩展 |
| PR 8（可观测性 / Replay） | `PluginInfo` 输出到 metrics；`evaluations_total` 等 runtime 数据上报 Prometheus |

---

## 下一步

1. **review 本 PR 3 详细文档**（重点：sub-task 拆分、6 行 cache 失效表、auth 分级、Q1-Q5 决议）
2. **确认 v1 不含 K8s SA / SPIFFE 是否可接受**——如果需要 v1 必含，估算 +2.5 天
3. **review 后启动 PR 3 实施**（建议双人并行；K8s SA / SPIFFE 推 PR 3.5）
