# DEP-XXXX PR 4 详细分解：类型感知合并 + chain-augment

> **状态**：v2.0 **CODED**（全部 6 个 sub-task 落地 / 2026-04-22）
> **依赖**：PR 1（proto + Pydantic types）
> **下游**：PR 5（orchestrator 在 pipeline 中调用 merge 函数）
> **预估工作量**：1 工程师 × 1-1.5 周；2 工程师并行 ~5-7 天

---

## 修订历史

### v2.0（2026-04-22）—— PR 4 全部 6 sub-task 实施完成

- **4-3 落地**：`tests/plugins/merge/test_type_aware_short_circuit.py`（9 cases）——REJECT 矩阵（单 REJECT / REJECT+SET / REJECT+final / 多 REJECT 首个 reason）+ final 矩阵（单 final / 多 final priority-smallest / final 覆盖 non-final bounds / final 仅 AT_LEAST / final in CONSTRAIN 含 SET）
- **4-4 落地**：`tests/plugins/merge/test_type_aware_worked_examples.py`——逐字复刻主文档 v11 PROPOSE 9 个 worked examples；使用 `PR` / `OR` / `CT` / `key` 助手让测试代码读起来像表；加 `test_worked_examples_count_matches_main_doc` 为 count 漂移 tripwire
- **4-5 落地**：`components/src/dynamo/planner/plugins/merge/chain_augment.py` + `tests/plugins/merge/test_chain_augment.py`（15 cases）—— 实现 priority-desc chain + `_partial_merge` 函数（`Optional[float]` None-preserves, non-None-overrides）+ runtime misuse detect (non-lowest priority + final=True → WARNING log + `misuse_warnings` append)
  - **⚠️ 与 v1.1 文档偏离**：PR 1 as-built 的 `PredictStageResponse` 只有 `predictions` / `reason` / `final` 三字段，**没有 REJECT 机制**。原 4-5 伪代码用 `resp.result` + `RejectResult` 是过时 draft。实施调整：`predictions=None` 即 ACCEPT，不 populate `degraded`（`ChainAugmentOutcome.degraded` 在 v1 永远为 `[]`）；README 已列为 future work：proto 未来加显式 REJECT 后再 populate
  - **接口调整**：使用 `PipelineContext.model_copy(update={"predictions": prediction})`（Pydantic v2 标准 API），而非 PR 4 伪代码中的 `with_predictions` 方法（PipelineContext 无此方法）
  - **PredictionData 字段 set/unset 判定**：Pydantic v2 下 `Optional[float] = None` 的 set 检查就是 `is not None`，比 `HasField` 更简洁（P2-2 review 残留就此简化）
- **4-6 落地**：`tests/plugins/merge/test_type_aware_properties.py`（6 invariants）+ `tests/plugins/merge/test_chain_augment_properties.py`（3 invariants）+ `plugins/merge/README.md`
  - Hypothesis 依赖通过 `pytest.importorskip("hypothesis")` guard；未安装时整个文件自动 skip。**Q4 pyproject/setup dev-extra 声明尚未 commit**（本地已 `pip install hypothesis==6.152.1` 验证通过）；merge 到 main 前建议把 `hypothesis>=6.0` 加到 `pyproject.toml` dev/test 依赖组
  - README 含 7 节：算法总览 / CONSTRAIN 模式 / worked example 指引 / chain_augment / final 语义差异表 / **chain-augment final 使用规范（强制契约）** / 测试布局 / future work
  - 性能 O(P×C) 注记（v1 不优化）

- **4-2 落地**：`components/src/dynamo/planner/plugins/merge/type_aware.py` + `tests/plugins/merge/{test_type_aware_basic.py, test_type_aware_constrain.py}` 已提交

- **4-2 落地**：`components/src/dynamo/planner/plugins/merge/type_aware.py` + `tests/plugins/merge/{test_type_aware_basic.py, test_type_aware_constrain.py}` 已提交
  - 按 PR 4 4-2 伪代码实现，含 5 条关键设计决策（REJECT > final；baseline 必填；final 路径 set_dropped 也记录；replicas=None 跳过；int(result) cast）
  - 输出顺序：**`plugin_results` 插入顺序 first，再 baseline-only keys**（非 set 迭代，保证跨 CPython 版本稳定）—— 原伪代码用 `set` 会非确定；调整为 dict 插入序 + 尾 append baseline-only
  - 基础测试 17 cases：空输入 / accept 透传 / SET / AT_LEAST / AT_MOST / clamp 优先序 / multi-component / multi-pool / unset 跳过 / baseline-only key 在输出
  - CONSTRAIN 测试 5 cases：SET drop + baseline 透传 / bounds 正常 / 混合 SET+bounds / 同 key 多 SET 每个都记录 set_dropped（给 orchestrator Prometheus counter 准确计数）
  - `__init__.py` 解开 `type_aware_merge` 的 re-export；剩 `chain_augment` TODO 给 4-5
- **最终验证（全部 6 sub-task 完成）**：
  - `pytest dynamo/planner/tests/plugins -q` → **186 passed**（121 baseline + 65 new: 22 basic/constrain + 9 short_circuit + 10 worked_examples + 15 chain_augment + 9 hypothesis properties）
  - CI-parity `pre_merge and planner and gpu_0` → **384 passed, 1 skipped, 11 deselected**（baseline 319 + 65 new）
  - proto stubs 无漂移

### v2.0 初段（2026-04-22）—— 4-1 实施完成

- **4-1 落地**：`components/src/dynamo/planner/plugins/merge/{__init__.py, types.py}` 已提交（无测试；纯定义）
  - `PluginResult` / `ComponentKey` 使用 `@dataclass(frozen=True)`；`ComponentKey` 依赖 hashable 作 4-2 桶 key
  - `MergeOutcome` / `ChainAugmentOutcome` 使用可变 `@dataclass`（含 `field(default_factory=list)` 的 `set_dropped` / `degraded` / `misuse_warnings`），便于 4-2 / 4-5 增量 append
  - 新增 `PredictPluginCallable`（`@runtime_checkable` `Protocol`）——原 4-5 接口签名中的 `list[PredictPluginCallable]` 类型占位；将 merge 模块与 PR 2/3 的具体 registry / transport 解耦
  - `PluginResult.result` 类型是 `Union[AcceptResult, OverrideResult, RejectResult]`，与主文档 v11 § G-2 对齐；注释说明 `final` 对 CONSTRAIN 静默忽略
- **__init__.py 临时形态**：4-1 仅 re-export `types` 的 5 个符号；`type_aware_merge` 和 `chain_augment` 的 re-export 分别在 4-2 / 4-5 落地时取消注释（代码内已有 TODO 注释标注）。原因：保持 baseline 可导入（`import dynamo.planner.plugins.merge` 今日可用，避免 in-progress 子任务残留破坏包导入）
- **验证**：121 baseline 测试全部通过；proto stubs 无漂移；`isinstance(Dummy, PredictPluginCallable)` runtime check OK；`PluginResult` 赋值触发 `FrozenInstanceError`

### v1.1（2026-04-20）—— Review 修订（方案 C：P0 + P1-2）
- **P1-2**：chain-augment final 语义升级为「强制契约」——final=true plugin 必须设 priority=最低数字（最高优先级）；4-5 chain_augment 算法加 runtime detect non-lowest-priority + final misuse；emit WARNING log + Prometheus metric；`ChainAugmentOutcome` 加 `misuse_warnings` 字段；4-6 README 加「chain-augment final 使用规范」专节
- **P0-1 关联**：跨 sub-task 协调点 §1 加强 PR 1 `PredictionData.optional` 必须项的 cross-ref
- 其他 review 残留（P2-2 `_is_set` 实现选择 / P2-3 hypothesis budget）记入 Implementation Breakdown 「PR 1-4 Review 残留问题」节，实施时 mitigation

### v1.0（2026-04-20）—— 初稿
- 与主文档 v10 对齐
- 6 个 sub-task：merge 目录 / type-aware merge / chain-augment / worked example test / property-based test / README
- **测试覆盖优先级**：worked example 表 + chain-augment 4 种用法是 must-pass；property-based test 用 hypothesis 验证不变量
- **设计原则**：纯函数模块 —— 无 I/O、无 state、无 Clock 依赖；任何函数 deterministic 可单独测试

---

## 为什么 PR 4 风险低

| 维度 | 评级 | 理由 |
|---|---|---|
| 与现有代码冲突 | **零** | 全部新建文件 |
| 算法复杂度 | **中** | type-aware 算法 4 步规则简单；但 final 与 REJECT 交互需要细致；chain-augment partial-merge 需要 careful |
| 测试可达 100% 覆盖 | **极高** | 纯函数 → property-based test + table-driven test 可达完美覆盖 |
| 阻塞下游 | **高** | PR 5 pipeline 必须用 PR 4 函数 |

**核心风险**：worked example 表与代码行为不一致 → 用户合并语义难懂。**唯一缓解**：4-4 子任务**逐字 copy v9 主文档表，作为参数化测试 source of truth**。

---

## 范围

**新建**：
- `components/src/dynamo/planner/plugins/merge/` 目录（type-aware + chain-augment）
- 单测 + property-based test
- README

**不动**：
- 不动 `core/` 任何文件
- 不依赖 PR 2 / PR 3（merge 是纯函数；不需要 Clock / Transport / Registry）
- **可与 PR 2 / PR 3 完全并行启动**

---

## 子任务清单（6 项）

### 4-1：merge 目录结构 + 内部数据类型

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/merge/__init__.py`<br/>`components/src/dynamo/planner/plugins/merge/types.py` |
| 关键数据类型 | <pre>@dataclass(frozen=True)<br/>class PluginResult:<br/>    """Single plugin's output for a stage, paired with its priority."""<br/>    plugin_id: str<br/>    priority: int<br/>    result: AcceptResult \| OverrideResult \| RejectResult<br/>    final: bool = False<br/><br/>@dataclass(frozen=True)<br/>class ComponentKey:<br/>    """Group key for type-aware merge buckets."""<br/>    sub_component_type: str<br/>    component_name: str \| None = None<br/><br/>@dataclass<br/>class MergeOutcome:<br/>    """Result of type_aware_merge."""<br/>    proposal: ScalingProposal \| None  # None on REJECT short-circuit<br/>    short_circuited: bool<br/>    short_circuit_reason: str = ""  # plugin_id of rejecter<br/>    used_final_from: str = ""       # plugin_id of final plugin (if any)<br/>    set_dropped: list[ComponentKey] = field(default_factory=list)<br/>      # CONSTRAIN-only: SET entries silently dropped (audit)<br/><br/>@dataclass<br/>class ChainAugmentOutcome:<br/>    """Result of chain_augment."""<br/>    prediction: PredictionData \| None<br/>    final_from: str = ""  # plugin_id that set final=true (if any)<br/>    degraded: list[str] = field(default_factory=list)  # plugins that REJECTed<br/>    misuse_warnings: list[str] = field(default_factory=list)  # P1-2 review:<br/>      # final=true plugins that were NOT lowest-priority (chain broke too early)</pre> |
| 测试 | 无（纯定义） |
| 依赖 | PR 1（OverrideResult / ComponentTarget / OverrideType / PredictionData） |
| 估算 | 0.25 天 |

---

### 4-2：type_aware_merge 主算法

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/merge/type_aware.py` |
| 接口 | <pre>def type_aware_merge(<br/>    plugin_results: list[PluginResult],<br/>    baseline: dict[ComponentKey, int],<br/>    set_allowed: bool = True,<br/>) -> MergeOutcome:<br/>    """Pure function. PROPOSE/RECONCILE/CONSTRAIN 三 stage 共用。<br/>    <br/>    Algorithm (v9 主文档 line 1246-1257):<br/>      1. REJECT 短路：任一 result 是 RejectResult → 整个返 short_circuited<br/>      2. final 优先：找 priority 最小的 final=true → 完全覆盖<br/>      3. 按 ComponentKey 分桶<br/>      4. 每桶内：floor=max(AT_LEAST), ceiling=min(AT_MOST),<br/>         recommendation=priority 最小的 SET 或 baseline,<br/>         result=clamp(recommendation, floor, ceiling)<br/>      <br/>    set_allowed=False → CONSTRAIN 模式：drop 任何 SET，<br/>    记录到 outcome.set_dropped 供 audit。<br/>    """<br/>    ...</pre> |
| **算法核心伪代码**（必须严格按主文档 v9）| <pre>def type_aware_merge(plugin_results, baseline, set_allowed):<br/>    # Step 1: REJECT 短路<br/>    for r in plugin_results:<br/>        if isinstance(r.result, RejectResult):<br/>            return MergeOutcome(<br/>                proposal=None,<br/>                short_circuited=True,<br/>                short_circuit_reason=f"{r.plugin_id}: {r.result.reason}",<br/>            )<br/>    <br/>    # Step 2: final 优先<br/>    overrides = [r for r in plugin_results <br/>                 if isinstance(r.result, OverrideResult)]<br/>    finals = [r for r in overrides if r.final]<br/>    if finals:<br/>        winner = min(finals, key=lambda r: r.priority)<br/>        targets = winner.result.targets<br/>        # In CONSTRAIN, drop SET even from final<br/>        if not set_allowed:<br/>            set_dropped = [<br/>                ComponentKey(t.sub_component_type, t.component_name)<br/>                for t in targets if t.type == OverrideType.SET<br/>            ]<br/>            targets = [t for t in targets if t.type != OverrideType.SET]<br/>        else:<br/>            set_dropped = []<br/>        return MergeOutcome(<br/>            proposal=ScalingProposal(targets=targets, source=winner.plugin_id),<br/>            short_circuited=False,<br/>            used_final_from=winner.plugin_id,<br/>            set_dropped=set_dropped,<br/>        )<br/>    <br/>    # Step 3-4: 按 key 分桶 + 类型合并<br/>    set_dropped = []<br/>    by_key: dict[ComponentKey, list[tuple[ComponentTarget, int]]] = {}<br/>    for r in overrides:<br/>        for t in r.result.targets:<br/>            key = ComponentKey(t.sub_component_type, t.component_name)<br/>            if t.type == OverrideType.SET and not set_allowed:<br/>                set_dropped.append(key)<br/>                continue<br/>            by_key.setdefault(key, []).append((t, r.priority))<br/>    <br/>    final_targets = []<br/>    all_keys = set(by_key.keys()) \| set(baseline.keys())<br/>    for key in all_keys:<br/>        entries = by_key.get(key, [])<br/>        at_least = [t.replicas for t, _ in entries <br/>                    if t.type == OverrideType.AT_LEAST and t.replicas is not None]<br/>        at_most = [t.replicas for t, _ in entries <br/>                   if t.type == OverrideType.AT_MOST and t.replicas is not None]<br/>        sets = [(t.replicas, prio) for t, prio in entries <br/>                if t.type == OverrideType.SET and t.replicas is not None]<br/>        floor = max(at_least) if at_least else 0<br/>        ceiling = min(at_most) if at_most else math.inf<br/>        if sets:<br/>            recommendation = min(sets, key=lambda x: x[1])[0]  # priority 最小<br/>        else:<br/>            recommendation = baseline.get(key, 0)<br/>        result_replicas = max(floor, min(ceiling, recommendation))<br/>        final_targets.append(ComponentTarget(<br/>            sub_component_type=key.sub_component_type,<br/>            component_name=key.component_name,<br/>            replicas=int(result_replicas),<br/>        ))<br/>    <br/>    return MergeOutcome(<br/>        proposal=ScalingProposal(targets=final_targets, source="merged"),<br/>        short_circuited=False,<br/>        set_dropped=set_dropped,<br/>    )</pre> |
| **关键设计决策** | <ol><li>**REJECT 优先于 final**——任何 REJECT 都立即短路，即使有 final 在场（一致与主文档 v9 line 1248；REJECT 是"安全否决"，final 是"权威覆盖"，安全否决更高）</li><li>**`baseline` 字段必填**——orchestrator 调用时传入"上一阶段输出"或"当前实际副本数"；无 SET 也无 plugin 涉及的 component_key 必须有 baseline 才能输出有意义的 ComponentTarget</li><li>**`set_dropped` 即使在 final 路径也记录**——CONSTRAIN final 含 SET 也被 drop（虽然 final 在 CONSTRAIN 被 silently ignored，但若 user 误用，drop SET 仍要 audit）</li><li>**`replicas=None`（unset）的 ComponentTarget 跳过**——主文档 v9 line 1078 「unset = 无意见」</li><li>**`int(result_replicas)`** —— floor/ceiling 可能是 inf 或 float，最终转 int；如果用户给 negative replicas，type-aware merge **不**做校验（pre-condition 由 plugin 保证）</li></ol> |
| 单测 | `tests/plugins/merge/test_type_aware_basic.py`：<br/>- 空 plugin_results + baseline → 返 ScalingProposal（baseline 透传）<br/>- 单 SET → 选中作为 recommendation<br/>- 多 SET 不同 priority → priority 最小者胜<br/>- 单 AT_LEAST → floor 生效<br/>- 多 AT_LEAST → 取 max<br/>- 单 AT_MOST → ceiling 生效<br/>- 多 AT_MOST → 取 min<br/>- AT_LEAST + AT_MOST 冲突（floor > ceiling）→ floor 胜（max(floor, min(ceiling, rec))，floor 在外层）<br/>- 多 component（prefill+decode 独立合并）<br/>- multi-pool（component_name 不同 = 不同桶）<br/><br/>`tests/plugins/merge/test_type_aware_constrain.py`：<br/>- `set_allowed=False` + plugin 给 SET → SET drop，set_dropped 列出该 key<br/>- `set_allowed=False` + plugin 给 AT_LEAST/AT_MOST → 正常合并 |
| 依赖 | 4-1, PR 1 |
| 估算 | 1.5 天 |

---

### 4-3：REJECT 与 final 路径单测

| 项 | 内容 |
|---|---|
| 实现位置 | `tests/plugins/merge/test_type_aware_short_circuit.py` |
| 测试矩阵 | <ol><li>**REJECT 短路**：<ul><li>单 REJECT → MergeOutcome.short_circuited=True，proposal=None</li><li>REJECT + 其他 SET → 仍短路（REJECT 优先）</li><li>REJECT + final → 仍短路（REJECT 优先于 final）</li><li>多 REJECT → short_circuit_reason 含第一个</li></ul></li><li>**final 优先**：<ul><li>单 final SET → 该 plugin OverrideResult 直接成为输出</li><li>多 final → priority 最小者胜</li><li>final + 其他 plugin AT_LEAST/AT_MOST → 其他被丢弃</li><li>final 输出无 SET 仅 AT_LEAST → 输出仅含该 AT_LEAST</li><li>final 在 CONSTRAIN 模式（set_allowed=False）+ 含 SET → SET drop + set_dropped 记录 + final 仍生效</li></ul></li></ol> |
| 依赖 | 4-2 |
| 估算 | 0.75 天 |

---

### 4-4：worked example 表逐字断言

| 项 | 内容 |
|---|---|
| 实现位置 | `tests/plugins/merge/test_type_aware_worked_examples.py` |
| 目的 | **逐字复刻 v9 主文档 PROPOSE 节中的 worked example 表**，作为 source of truth |
| 测试矩阵（来自 v9 主文档 PROPOSE 节，**逐字 copy**）| <pre>WORKED_EXAMPLES = [<br/>    # (case_name, plugin_results, baseline, expected_target_replicas)<br/>    # 单 component 5 行：<br/>    ("only_baseline", [], {key("prefill"): 5}, {key("prefill"): 5}),<br/>    ("only_set",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 8)]))],<br/>     {key("prefill"): 5},<br/>     {key("prefill"): 8}),<br/>    ("set_priority_wins",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 8)])),<br/>      PR("p2", 50, OR([CT("prefill", SET, 10)]))],<br/>     {key("prefill"): 5},<br/>     {key("prefill"): 10}),  # p2 priority 更小（数字小）<br/>    ("set_with_at_least_floor",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 4)])),<br/>      PR("p2", 50, OR([CT("prefill", AT_LEAST, 6)]))],<br/>     {key("prefill"): 5},<br/>     {key("prefill"): 6}),  # SET=4 被 floor=6 顶高<br/>    ("set_with_at_most_ceiling",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 12)])),<br/>      PR("p2", 50, OR([CT("prefill", AT_MOST, 8)]))],<br/>     {key("prefill"): 5},<br/>     {key("prefill"): 8}),  # SET=12 被 ceiling=8 压低<br/>    # 多 component 2 行：<br/>    ("multi_component_independent",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 8), CT("decode", SET, 4)]))],<br/>     {key("prefill"): 5, key("decode"): 3},<br/>     {key("prefill"): 8, key("decode"): 4}),<br/>    ("multi_component_mixed_types",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 8), CT("decode", AT_MOST, 6)])),<br/>      PR("p2", 50, OR([CT("decode", SET, 10)]))],<br/>     {key("prefill"): 5, key("decode"): 3},<br/>     {key("prefill"): 8, key("decode"): 6}),  # decode SET=10 被 AT_MOST=6 压低<br/>    # 分层多池（hierarchical pools）1 行：<br/>    ("hierarchical_pools",<br/>     [PR("p1", 100, OR([<br/>          CT("prefill", "pool-A", SET, 8),<br/>          CT("prefill", "pool-B", SET, 4),<br/>      ]))],<br/>     {key("prefill", "pool-A"): 5, key("prefill", "pool-B"): 3},<br/>     {key("prefill", "pool-A"): 8, key("prefill", "pool-B"): 4}),<br/>    # final 覆盖 1 行（v6 新增）：<br/>    ("final_override_completely",<br/>     [PR("p1", 100, OR([CT("prefill", SET, 8)]), final=True),<br/>      PR("p2", 50, OR([CT("prefill", SET, 10)])),<br/>      PR("p3", 30, OR([CT("prefill", AT_MOST, 4)]))],<br/>     {key("prefill"): 5},<br/>     {key("prefill"): 8}),  # p1 final 完全覆盖；p2/p3 全部丢弃<br/>]<br/><br/>@pytest.mark.parametrize("case_name,prs,base,expected", WORKED_EXAMPLES, ids=lambda c: c[0] if isinstance(c, tuple) else "")<br/>def test_worked_example(case_name, prs, base, expected):<br/>    out = type_aware_merge(prs, base, set_allowed=True)<br/>    actual = {ComponentKey(t.sub_component_type, t.component_name): t.replicas <br/>              for t in out.proposal.targets}<br/>    assert actual == expected, f"case={case_name}: expected={expected}, got={actual}"</pre> |
| **关键约束** | <ol><li>**任何 v9 主文档 worked example 表的修改必须同 PR 改本测试**——主文档与测试是 lock-step；用一个 git pre-commit hook 阻止仅改一边</li><li>**测试 ID 必须与 case_name 一致**——CI 输出可读，定位 fail 容易</li><li>**助手 ctor**：`PR(plugin_id, priority, result, final=False)`、`OR(targets)`、`CT(...)` 在 test fixture 中定义，让测试代码读起来像主文档表 |
| 依赖 | 4-2 |
| 估算 | 1 天 |

---

### 4-5：chain_augment 主算法 + 4 种用法测试

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/merge/chain_augment.py`<br/>`tests/plugins/merge/test_chain_augment.py` |
| 接口 | <pre>async def chain_augment(<br/>    plugin_chain: list[PredictPluginCallable],<br/>    initial_context: PipelineContext,<br/>) -> ChainAugmentOutcome:<br/>    """Pure function (modulo plugin_chain async calls).<br/>    <br/>    Algorithm (v9 主文档 line 1294-1313):<br/>      1. plugins 按 priority 降序排（数字大者先；最高 priority/数字小最后）<br/>      2. prediction = None<br/>      3. 顺序调每个 plugin，传当前 prediction 进 context<br/>         - AcceptResult → prediction 不变<br/>         - RejectResult → degraded[+plugin_id]，prediction 不变<br/>         - PredictStageResponse → partial-merge 进当前 prediction<br/>         - response.final=True → 立即 break<br/>    """<br/>    chain = sorted(plugin_chain, key=lambda p: -p.priority)<br/>    lowest_prio = min((p.priority for p in chain), default=None)<br/>    prediction: PredictionData \| None = None<br/>    final_from = ""<br/>    degraded = []<br/>    misuse_warnings: list[str] = []  # P1-2 review: runtime-detected misuse<br/>    <br/>    for p in chain:<br/>        ctx = initial_context.with_predictions(prediction)<br/>        resp = await p.call("Predict", ctx)<br/>        if isinstance(resp.result, AcceptResult):<br/>            continue<br/>        if isinstance(resp.result, RejectResult):<br/>            degraded.append(p.plugin_id)<br/>            continue<br/>        prediction = _partial_merge(prediction, resp.predictions)<br/>        if resp.final:<br/>            final_from = p.plugin_id<br/>            # P1-2 review: detect non-lowest-priority final misuse<br/>            if p.priority != lowest_prio:<br/>                msg = (f"chain_augment_final_misuse: plugin_id={p.plugin_id} "<br/>                       f"priority={p.priority} returned final=true but is NOT "<br/>                       f"lowest priority (={lowest_prio}). Risk: chain broke "<br/>                       f"BEFORE higher-priority plugin ran. See PR 4 README "<br/>                       f"'chain-augment final 使用规范'.")<br/>                log.warning(msg)<br/>                misuse_warnings.append(msg)<br/>            break<br/>    <br/>    return ChainAugmentOutcome(<br/>        prediction=prediction,<br/>        final_from=final_from,<br/>        degraded=degraded,<br/>        misuse_warnings=misuse_warnings,  # PR 5 orchestrator emits Prometheus<br/>                                          # predict_chain_final_at_non_lowest_priority_total<br/>    )<br/><br/>def _partial_merge(prev: PredictionData \| None, new: PredictionData) -> PredictionData:<br/>    """Field-level merge: new field overrides prev field if set;<br/>       unset (default zero) preserves prev."""<br/>    if prev is None:<br/>        return new<br/>    return PredictionData(<br/>        predicted_num_req=new.predicted_num_req if _is_set(new, "predicted_num_req") else prev.predicted_num_req,<br/>        predicted_isl=new.predicted_isl if _is_set(new, "predicted_isl") else prev.predicted_isl,<br/>        predicted_osl=new.predicted_osl if _is_set(new, "predicted_osl") else prev.predicted_osl,<br/>        source=new.source or prev.source,<br/>    )</pre> |
| **关键设计决策** | <ol><li>**partial-merge 的"set vs unset"判定**：proto3 default 是 0.0；当 plugin 输出 `predicted_num_req=0.0` 时，无法区分"我说 0"和"未设置"。**v1 方案**：用 explicit "all-or-nothing" —— plugin 必须输出完整三元组，否则视为未矫正（保留 prev）。但这与主文档 v9 line 1320 "user-llm-predictor 输出 (num_req=1200) 其余字段保留" 矛盾<br/>**修订**：proto 加 `optional` 修饰（PR 1 1-3 修订）—— 每个 PredictionData 字段加 `optional`；then `_is_set` 用 `HasField`。**本 PR 4 假设 PR 1 已加 optional**；如果 PR 1 没加，4-5 子任务**必须先 patch PR 1**</li><li>**chain 排序：priority 降序**——数字大的先（低优先级），数字小的后（高优先级）；最后跑的 plugin 拍板（"高优先级最后说"）。这与 PROPOSE 的 priority 规则一致（priority 数字小 = 高优先级）</li><li>**REJECT 不短路 chain**——与 type-aware merge 的 REJECT 不同；PREDICT 的 REJECT 仅"自己缺席"，degraded 列表给 audit 用</li><li>**`initial_context` immutable**——chain_augment 不修改入参；每次循环 `ctx = initial_context.with_predictions(prediction)` 构造新 context（PipelineContext 应实现 `with_predictions` 等 immutable update）</li></ol> |
| 单测（4 种用法 × 多 case）| <ol><li>**Replace（最简单）**：单 plugin 输出完整 PredictionData → final 等于该输出</li><li>**Patch（v9 line 1320 例子）**：<ul><li>低 prio plugin 输出 (1000, 3000, 150)</li><li>高 prio plugin 输出 (num_req=1200) 其余字段保留</li><li>最终：(1200, 3000, 150)</li></ul></li><li>**Augment**（多 plugin 各自补充字段）：<ul><li>plugin A 输出 (num_req=1000)</li><li>plugin B 输出 (isl=3000, osl=150)</li><li>最终：(1000, 3000, 150)</li></ul></li><li>**Passthrough**：<ul><li>所有 plugin 返 ACCEPT</li><li>prediction 保持 None；下游 PROPOSE 走默认</li></ul></li><li>**REJECT 不影响 chain**：<ul><li>plugin A 输出 (1000, 3000, 150)</li><li>plugin B 返 REJECT → degraded=[B]，prediction=(1000,3000,150)</li><li>plugin C 输出 (num_req=1500)</li><li>最终：(1500, 3000, 150)</li></ul></li><li>**final 提前终止**：<ul><li>3 plugin chain，第 2 个 final=true</li><li>第 3 个 plugin **不被调用**（mock 验证 call_count=0）</li><li>final_from = plugin_2.id</li></ul></li><li>**多 final**：<ul><li>chain 里 plugin A (prio=10, final=true)、plugin B (prio=5, final=true)</li><li>因 chain 排序 prio 降序，B (prio=5) 最后跑</li><li>A 触发 break；B **不被调用**——`final_from=plugin_A.id`</li><li>**这与 PROPOSE 的 final 规则不同**（PROPOSE 是所有 plugin 都跑、合并时 priority 最小者胜；chain 内 chain 顺序决定胜者，而 chain 排序保证后跑的是高 priority/数字小，所以"先到的 final" = 较低 priority 的 final，违反"priority 最小者胜"直觉）<br/>**结论**：chain-augment 的 final 是**第一个出现 final=true 的胜**——即"最低 priority 的 final"，而非"priority 最小（最高）的 final"。**这是 chain 顺序的副作用**，4-5 README 必须明确</li></ul></li></ol> |
| 依赖 | 4-1, PR 1（PredictionData 含 optional 字段；如未加，本 sub-task 先 patch PR 1） |
| 估算 | 2 天 |

---

### 4-6：property-based test (hypothesis) + README

| 项 | 内容 |
|---|---|
| 实现位置 | `tests/plugins/merge/test_type_aware_properties.py`<br/>`tests/plugins/merge/test_chain_augment_properties.py`<br/>`components/src/dynamo/planner/plugins/merge/README.md` |
| Property-based 测试 | 用 `hypothesis` 库（dynamo 是否已用？需调研；如未用，PR 4 引入）：<br/>**type-aware merge 不变量**：<ol><li>**Monotonicity**：加 plugin 只会让结果更紧——任意添加一个 AT_LEAST plugin，结果 ≥ 原结果；任意添加 AT_MOST plugin，结果 ≤ 原结果</li><li>**Idempotency**：merge(prs, base) == merge(prs + [duplicate of any pr], base)（重复 plugin 无影响）</li><li>**REJECT dominance**：merge(prs + [REJECT], base) 永远 short_circuited</li><li>**final dominance**：merge(prs + [final SET=k, prio=0], base)（priority 0 是最小）始终 returns SET=k 的 proposal（除非有 REJECT）</li><li>**clamp correctness**：所有输出 replicas 都满足 max(0, floor) ≤ replicas ≤ min(inf, ceiling)</li><li>**baseline preservation**：无 plugin 输出 → all targets = baseline</li></ol><br/>**chain-augment 不变量**：<ol><li>**Sort determinism**：相同 priority → 顺序由 plugin_id 决定（次序稳定）</li><li>**Final break**：chain 里第一个 final 之后的 plugin 永不调用（mock call_count）</li><li>**REJECT preservation**：REJECT 不破坏 prediction state；degraded 准确记录</li></ol> |
| README 内容 | <ol><li>type-aware merge 算法 4 步骤总览（与主文档 v9 line 1246-1257 对齐）</li><li>chain-augment 算法（与主文档 v9 line 1294-1313 对齐）</li><li>**worked example 表**（直接引用主文档；本 README 不重复，给链接 + 锚点）</li><li>**关于 final 的两种语义对照**：type-aware（priority 最小者胜）vs chain-augment（chain 中第一个 final 胜，即低 priority 的 final）</li><li>**【强制契约 — P1-2 review】「chain-augment final 使用规范」专节**：<ul><li>规则：**final=true 的 PREDICT plugin 必须设 priority=最低数字（最高优先级）**</li><li>反例：mid-prio plugin (priority=100, final=true) + emergency plugin (priority=5) → emergency 永不被调用</li><li>正例：emergency-override plugin 设 priority=5 + final=true → chain 末位运行 → 它的输出胜</li><li>误用监测：chain_augment 在 runtime detect non-lowest priority + final=true → emit `WARNING` log + Prometheus `predict_chain_final_at_non_lowest_priority_total{plugin_id}` 计数器（PR 8 加 alert rule）</li><li>修复指引：发现 alert 后，把 final plugin 的 priority 降到当前 chain 中最低数字，或者把 final 改 `final=false`（让算法走 chain-augment 而非 break）</li></ul></li><li>使用建议：何时用 type-aware（PROPOSE/RECONCILE/CONSTRAIN）何时用 chain-augment（PREDICT）</li><li>性能注意：type-aware O(P × C)（P=plugins, C=components）；chain-augment O(P) 串行 await</li></ol> |
| 依赖 | 4-2, 4-3, 4-4, 4-5 |
| 估算 | 1.5 天（property test 1 天 + README 0.5 天） |

---

## PR 4 总估算

- **单人**：~7 天（4-1: 0.25 + 4-2: 1.5 + 4-3: 0.75 + 4-4: 1.0 + 4-5: 2.0 + 4-6: 1.5 = 7.0 天）
- **双人并行**：~4-5 天
  - 工程师 A：4-1 → 4-2 → 4-3 → 4-4（type-aware 主路径）
  - 工程师 B：4-5（chain-augment 独立）→ 4-6（合并 property test + README）
  - 汇合时间：4-2 完成后 4-6 才能合并 type-aware property test（4-2 → 4-6 串行）

注：原 Implementation Breakdown 估算 1 工程师 × 1-1.5 周（≈ 5-7.5 天）合理。本 PR 4 详细文档建议按 **单人 7 天 / 双人 4-5 天** 排期。

---

## PR 4 Acceptance Criteria

- [ ] `type_aware_merge` 实现，所有单测通过（basic / constrain / short_circuit）
- [ ] **worked example 表 9 个 case 全部通过**——逐字与主文档 v9 PROPOSE 节对齐
- [ ] `chain_augment` 实现，4 种用法（replace / patch / augment / passthrough）+ REJECT + final 单测全部通过
- [ ] Property-based test：type-aware merge 6 个不变量 + chain-augment 3 个不变量 全部通过
- [ ] `set_allowed=False` 在 type-aware merge 正确丢弃 SET 并填 set_dropped 列表
- [ ] CI `planner-test` job 自动包含全部测试
- [ ] PR description 明确：本 PR **零 production 影响**（纯函数模块、无调用方）
- [ ] PR description 包含：worked example 表的快照（链接到主文档 + 测试 ID 对照表）
- [ ] **chain_augment runtime lint warning** 实现（detect "non-lowest priority + final" → 一次性 WARNING log，P1-2 review 决议）
- [ ] **README 含「chain-augment final 使用规范」专节**，明确 final=true plugin 必须设 priority=最低数字（P1-2 review 决议）

---

## 跨 Sub-task 必须协调的点

### 1. PR 1 必须先确认 `PredictionData` 字段是 `optional`

4-5 sub-task 的 partial-merge 强依赖 proto3 `optional` field（用 `HasField` 区分 set vs default-zero）。**PR 4 启动前必须 verify PR 1 的 `PredictionData` 定义**：

```proto
message PredictionData {
  optional float predicted_num_req = 1;  // ← MUST be optional
  optional float predicted_isl     = 2;
  optional float predicted_osl     = 3;
  string source                    = 4;
}
```

如果 PR 1 没加 `optional`，**4-5 子任务的第一步是先 patch PR 1**——这是 inline correction，不算 PR 4 范围扩张。

### 2. type-aware 与 chain-augment 的 final 语义差异 —— 必须强制契约（P1-2 review 升级）

| stage | final 胜出规则 | 用户契约 |
|---|---|---|
| PROPOSE / RECONCILE | priority 数字最小（即最高 priority）的 final 胜 | 任意 priority 都可设 final |
| PREDICT (chain-augment) | chain 中**第一个**出现 final=true 的胜——chain 排序后 priority 数字**大**的（较低 priority）的 final 先 break | **强制契约**：final=true plugin 必须设 priority=**最低数字**（即最高 priority）；否则会被低 priority plugin 抢 break |

**为什么 chain-augment final 是「巧合的」正确**：

chain 按 priority 降序排（数字大者先跑），所以「最高 priority」（数字最小）的 plugin 永远在 chain **最后**位置。常见 use case 是 emergency-override plugin 设最高 priority + final=true → 它最后跑、它的 final 触发 break、它的输出作为最终 prediction → ✓ 直觉对。

但如果用户写 `mid-priority plugin (priority=100, final=true)` + `emergency plugin (priority=5, final=false)`，chain 顺序是 `[priority=100, priority=5]` → priority=100 先跑、final 触发 break → priority=5 **永远不被调用** → ✗ 反直觉（用户预期「emergency 是更高优先级，应该胜」）。

**强制契约（v1）**：

1. **4-6 README** 必须有专节「chain-augment final 使用规范」，明确：「**final=true 的 PREDICT plugin 必须设 priority=最低数字（最高优先级）**」
2. **runtime lint warning**（4-5 sub-task 实现）：chain_augment 启动时 detect "non-lowest priority + final" 组合，emit `WARNING` log 一次性提示用户（不阻止运行，但 audit）
3. **PR 5 orchestrator 加 metric**：`predict_chain_final_at_non_lowest_priority_total` ——运维可据此发现误用
4. **未来 v2 考虑**：chain_augment 在 sort 时把 `final=true && priority=非最小` 的 plugin 强制放到 chain 末尾——但这破坏「priority 降序」简单规则，v1 不做（YAGNI）

### 3. `_partial_merge` 的字段 set vs unset 判定

仅用于 `chain_augment._partial_merge`，与 type-aware merge 无关。但其他 stage 如果未来引入"partial output"，需要复用 _partial_merge 思路——4-6 README 列为 future-proof note。

### 4. baseline 默认值由 orchestrator 计算，PR 4 不假设

`type_aware_merge` 的 `baseline` 字段是必填 dict；空 dict 等价于"所有 component 都没 baseline"。**orchestrator** (PR 5) 负责构造 baseline：
- PROPOSE 阶段：baseline = current worker counts
- RECONCILE 阶段：baseline = PROPOSE 输出
- CONSTRAIN 阶段：baseline = RECONCILE 输出

PR 4 不实现 baseline 计算逻辑，**只信任传入值**。

---

## 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| worked example 表与代码偏离 → 用户合并语义难懂 | **中** | 4-4 子任务 9 个 case 是 must-pass；任何 worked example 修改必须同 PR 改测试 |
| `_partial_merge` proto3 optional 判定错 → patch 类型 plugin 输出被丢 | **中** | 4-5 sub-task 单测覆盖每个字段单独 set vs unset；PR 1 必须先确认 optional |
| Property-based test 找到 corner case → CI flaky | **低-中** | hypothesis 默认 100 example；CI 设 deadline；如发现 corner case 必须修代码或测试，**禁止** seed 跳过 |
| chain-augment final 语义反直觉（"低 priority 先 final 胜"）→ 用户 plugin 行为意外 | **中** | 4-6 README 明确说明；4-5 测试 case 6 显式覆盖；后续考虑给 chain-augment 加 `priority_strict_final` toggle 让用户选 |
| 性能：高 plugin 数 + 高 component 数 → type-aware merge O(P × C) 太慢 | **低** | 当前生产 P ≤ 10、C ≤ 5；实测 < 1ms；不需要预先优化；如果未来 P 大幅增长，加 ComponentKey-keyed 索引 |

---

## 推荐 staffing

- **1 名 Backend / 算法工程师**（清晰函数式思维）：负责 4-1 / 4-2 / 4-3 / 4-4
- **1 名 Backend / 测试工程师**（hypothesis / pytest 经验）：负责 4-5 / 4-6

---

## Resolved Questions（已决议）

### Q1：type_aware_merge 是否需要支持「baseline by ComponentTarget」（不止数字）？

**决议**：**否**（v1）。理由：
- baseline 当前只是数字（int）；如未来需要"上一阶段输出的 ComponentTarget"作为 baseline（含 type 字段），那时再加新参数 `baseline_targets: list[ComponentTarget]`
- 简化 v1 接口，避免过度设计

### Q2：`set_allowed=False` 时 SET 是否记 audit 而非完全丢弃？

**决议**：**完全丢弃 + set_dropped 记录用 audit**。理由：
- 主文档 v9 line 1142-1148 明确"runtime drop + audit"；与 PR 4 实现一致
- `set_dropped` 列表给 orchestrator emit Prometheus metric `plugin_constrain_set_dropped_total`

### Q3：`chain_augment` 是否要在每次 merge 后回写 `PipelineContext`？

**决议**：**否**。`chain_augment` 是纯函数；返回 `prediction` 由 orchestrator (PR 5) 决定写到哪里。理由：
- 让 chain_augment 不依赖 PipelineContext mutator（保持纯函数）
- orchestrator 可能选择在 chain 结束后才写 context（一次性）vs 每步写（debug 模式）

### Q4：是否引入 `hypothesis` 作为 dynamo 新依赖？

**决议**：**是，但仅在 dev/test 依赖中**。理由：
- property-based test 价值高（catch corner case）；其他 dynamo 模块已有需求
- 在 `setup.py` 加 `extras_require={"dev": ["hypothesis>=6.0"]}`，不影响 production install

### Q5：`type_aware_merge` 是否要 async？

**决议**：**否**。纯函数无 I/O，sync 即可。理由：
- async 仅在调 plugin（PR 5 orchestrator 做）；merge 自身不涉及 I/O
- sync 函数可在测试中直接调，无 event loop 启动开销

---

## 已删除的内容（v0 → v1）

无（首版）

---

## 与其他 PR 的接口

| 下游 PR | 依赖 PR 4 的内容 |
|---|---|
| PR 5（Orchestrator） | `type_aware_merge` 在 PROPOSE/RECONCILE/CONSTRAIN 三 stage 调用；`chain_augment` 在 PREDICT 调用；`MergeOutcome` 含 `short_circuited` 用作 EXECUTE skip 决策 |
| PR 6（5 个 builtin） | `OverrideResult` / `ComponentTarget` / `OverrideType` 是 builtin plugin 输出；与 PR 4 接口一致 |
| PR 8（可观测性） | `MergeOutcome.set_dropped` → `plugin_constrain_set_dropped_total` Prometheus metric |

---

## 下一步

1. **review 本 PR 4 详细文档**（重点：worked example 表完整性、chain-augment final 语义、Q1-Q5 决议）
2. **verify PR 1 `PredictionData` 字段是 `optional`**——如不是，PR 1 修订（约 0.25 天）
3. **review 后启动 PR 4 实施**（建议双人并行：A 走 type-aware；B 走 chain-augment + 收尾）
