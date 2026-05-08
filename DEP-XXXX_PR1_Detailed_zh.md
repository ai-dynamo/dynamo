# DEP-XXXX PR 1 详细分解：Proto + Type 骨架（无依赖）

> **状态**：DRAFT v1.0 — 2026-04-20
> **依赖**：无（所有后续 PR 的基石）
> **下游**：PR 2 / 3 / 4 / 5 / 6 / 7 / 8 全部 import PR 1 的 proto types
> **预估工作量**：1-2 工程师 × 2-3 天（单人 ~3 天；双人并行 ~1.5 天）

---

## 修订历史

### v2.0（2026-04-22）—— PR 1 实施完成 + 偏差同步

**实施状态**：PR 1 全部 9 sub-task 编码完成；27 个 round-trip 测试全部通过；`tools/build/gen_planner_proto.sh` + `--check` 模式工作正常。

**与 v1.1 文档的实施偏差**（按发现顺序）：

| 偏差点 | v1.x 文档 | 实际实施 | 决策理由 |
|---|---|---|---|
| proto 路径 | 顶层 `proto/dynamo/planner/plugin/v1/` | `components/src/dynamo/planner/plugins/proto/v1/` | 贴合 dynamo 现有 convention（`lib/llm/src/grpc/protos/` 也是 src 内）；省去 namespace package import path 配置 |
| 工具链 | 优先 buf，fallback grpc_tools | grpc_tools.protoc 直接 | dynamo 没有 buf.yaml；引入新工具链非 PR 1 范围；`grpcio-tools<=1.76.0` 已 pin |
| 编译入口 | `proto/dynamo/planner/plugin/v1/` | `-I components/src` 让 generated import 走 namespace path `dynamo.planner.plugins.proto.v1.plugin_pb2` | 同上 path 决策 |
| Pydantic 镜像 oneof | 简单 Optional 三选一 | 显式 `result_kind` tag + 三 payload field + `model_post_init` validator | 最准确镜像 proto3 oneof 语义；`_StageOneofResponse` 共享基类避免重复 |
| `_proto_bridge.py` 转换实现 | "通过 `model_dump()` + `proto_cls(**dict)` + `MessageToDict`" | 4 处 edge case 修复后才工作 | edge case 见下方「实施 footnote」 |

**实施 footnote（_proto_bridge 4 处必需 edge case）**：

1. `MessageToDict` 默认 enum 输出字符串名（"AT_LEAST"），Pydantic IntEnum 期待 int → 必须 `use_integers_for_enums=True`
2. Pydantic `mode="json"` 把 `bytes` UTF-8 decode 为 string，`\xff` 字节抛 UnicodeDecodeError → 改 `mode="python"` + 手动 `_normalize` walker base64 编码 bytes
3. proto map 字段在 `ListFields()` 返回 Python dict（非 list）；descriptor recurse 路径必须 detect 并跳过（map values 是 scalar）
4. proto repeated message 字段返回 `RepeatedCompositeContainer`，没有 `__iter__` 属性但能用 `iter()`；推导 child message 类型用 `iter(value)` 而非 `hasattr(value, "__iter__")`

**与下游 PR 的契约保持不变**：
- 所有 proto schema 决议（含 PredictionData optional / CONSTRAIN SET runtime drop / final 语义）按 v11 主文档实现；
- `dynamo.planner.plugins.types` namespace 导出 31 个 Pydantic class + 3 个 IntEnum，与 PR 4 / PR 5 / PR 6 引用一致；
- `_PYD_TO_PROTO` 字典含 33 个 message 双向映射，`test_class_coverage_*` 守护两端字段同步。

**新增（v2.0 实施补全）**：
- `dynamo.planner.plugins._proto_bridge` —— 公共工具类，给 PR 2 InProcessTransport / PR 5 Pipeline driver 复用
- `tests/plugins/proto/test_round_trip.py` —— 27 case，CI 已通过 marker 自动接入 `planner-test` job
- `proto/v1/README.md` —— 演化策略 + 关键 invariants + adding-new-message workflow

### v1.1（2026-04-20）—— Review 修订（方案 C：P0 + P1-2）
- **P0-1**：1-3 review point 加强制项「`PredictionData` 字段必须 `optional`」——PR 4 chain-augment partial-merge 的 hard contract
- 其他 review 残留（P1-1 / P1-3 / P1-5 / P2-1 / P2-2）记入 Implementation Breakdown 「PR 1-4 Review 残留问题」节，实施时 mitigation

### v1.0（2026-04-20）—— 初稿
- 与主文档 v10、Implementation Breakdown 对齐
- 修正 `PluginLifecycle` RPC 数量：v10 只剩 `Bootstrap` / `Reset`（YAGNI 删除 `Snapshot` / `Restore`）—— Implementation Breakdown 第 83 行需同步修
- 9 个 sub-task：proto 切分为 5 段（registry / pipeline / stage / lifecycle / 共享 enum），加 build script、Pydantic 镜像、round-trip test、README
- 明确 PR 1 不接业务逻辑：纯定义工作 + Pydantic 镜像 + 编译/契约测试

---

## 为什么 PR 1 风险低

| 维度 | 评级 | 理由 |
|---|---|---|
| 与现有代码冲突 | **极低** | 全部新建文件；零修改 |
| Acceptance 难度 | **低** | 编译通过 + round-trip 测试通过即合格 |
| 设计变更可能性 | **中** | proto 是契约，定义后 backward-incompatible 改动代价大 → 必须 review 仔细 |
| 阻塞下游 | **高** | 所有后续 PR 都 import；PR 1 不 ship，PR 2-8 全部不能启动 |

**核心风险**：proto 定义错了，后续 PR 全部要 patch（proto3 加新字段是 backward-compatible，但删字段、改 tag、改类型都不是）。**唯一缓解**是 review 阶段把 proto schema 当 API 来审。

---

## 范围

**新建**：proto schema 文件 + Python types 镜像 + 编译脚本 + 契约测试 + README。

**不动**：
- 不动现有 `core/` / `connectors/` / `monitoring/` / `tests/` 任何文件
- 不实现任何业务逻辑（Register / Tick / Merge 全部留给后续 PR）
- 不接 K8s / not 接 RPC server / not 接 transport（PR 2 / 3 才做）

---

## 子任务清单（9 项）

### 1-1：proto 目录结构 + 骨架

| 项 | 内容 |
|---|---|
| 新建 | `proto/dynamo/planner/plugin/v1/plugin.proto`（仅 package + syntax + 文件级 comment + 留空 service / message 占位）<br/>`proto/dynamo/planner/plugin/v1/README.md`（schema 演化策略：proto3 backward-compatible 规则、tag 不可重用、字段不可删等）<br/>`components/src/dynamo/planner/plugins/__init__.py`（空）<br/>`components/src/dynamo/planner/plugins/proto_gen/__init__.py`（空，等 1-6 生成代码注入） |
| 修改 | 无 |
| 测试 | 无（纯目录） |
| 依赖 | 无 |
| 估算 | 0.25 天 |

**proto 文件头（统一模板）**：

```proto
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Plugin contract for Dynamo Planner Plugin Architecture (DEP-XXXX).
// See DEP-XXXX_Dynamo_Planner_Plugin_Architecture_zh.md for full design.
//
// **Schema evolution policy** (proto3, must-follow):
//   1. NEVER reuse a field tag — adding `reserved` for any deleted tag.
//   2. NEVER change the type of an existing field.
//   3. NEVER rename an existing field (clients may key on field names in
//      reflection / json transcoding).
//   4. ALL new fields MUST be optional or have safe-zero defaults.
//   5. Bumping `protocol_version` (RegisterRequest.protocol_version) is
//      reserved for *additive* contract changes; *breaking* changes
//      require a new package path (v2/).

syntax = "proto3";

package dynamo.planner.plugin.v1;
```

---

### 1-2：PluginRegistry service + 关联 messages

| 项 | 内容 |
|---|---|
| 实现位置 | `proto/dynamo/planner/plugin/v1/plugin.proto`（在 1-1 骨架基础上加注） |
| 范围 | 1 service（4 RPC）+ 9 message + 2 enum：<br/>**Service**：`PluginRegistry { Register / Heartbeat / Unregister / ListPlugins }`<br/>**Messages**：`RegisterRequest`、`RegisterResponse`、`HeartbeatRequest`、`HeartbeatResponse`、`UnregisterRequest`、`UnregisterResponse`、`ListPluginsRequest`、`ListPluginsResponse`、`PluginInfo`<br/>**Enums**：`HoldPolicy { ACCEPT_WHEN_IDLE / HOLD_LAST }`、`CircuitState { CLOSED / OPEN / HALF_OPEN }` |
| 字段细节来源 | 主文档 v10 line 826-933（逐字段 copy，含全部 inline comment） |
| **关键 review point** | <ol><li>`RegisterRequest.needs` (`repeated string`) ——「length 0 with default = send full context」语义在主文档定，proto comment 必须保留这条说明</li><li>`RegisterRequest.fpm_encoding` 默认 "msgspec"，仅允许 `"msgspec" \| "proto" \| "json"`——comment 写清楚但不在 proto 强制（proto3 没有 enum value 校验，这是 server 侧职责）</li><li>`RegisterRequest.request_timeout_seconds = 0` 表示用 orchestrator default，**不是** 0 秒</li><li>`HoldPolicy` 默认值是 0 = `ACCEPT_WHEN_IDLE`——和主文档 v10 一致；如果想默认 `HOLD_LAST` 需调换 enum 顺序</li></ol> |
| 测试 | 无（纯定义；编译通过即可，编译留 1-6） |
| 依赖 | 1-1 |
| 估算 | 0.5 天 |

---

### 1-3：PipelineContext + 共享数据类型 messages

| 项 | 内容 |
|---|---|
| 实现位置 | `proto/dynamo/planner/plugin/v1/plugin.proto` |
| 范围 | 6 messages：<br/>`PipelineContext`、`ObservationData`、`TrafficMetrics`、`FpmData`、`WorkerState`、`PredictionData`、`ScalingProposal`、`ComponentTarget`、`OverrideResult`、`AcceptResult`、`RejectResult`<br/>2 enum：`OverrideType { SET / AT_LEAST / AT_MOST }`<br/>(共 11 message + 1 enum，但 ScalingProposal/ComponentTarget/OverrideResult/AcceptResult/RejectResult 给 1-4 stage service 复用) |
| 字段细节来源 | 主文档 v10 line 942-1096 |
| **关键 review point** | <ol><li>`FpmData.prefill_engines` / `decode_engines` 是 `map<string, bytes>`，**bytes 解释由 RegisterRequest.fpm_encoding 决定**——comment 必须保留这条 dispatch 规则</li><li>`PipelineContext.request_id` vs `decision_id` 的语义区分：每个 tick 都有 request_id，只有产生 proposal 时才有 decision_id（见主文档 v9 第 1024-1043 行）</li><li>`ComponentTarget.sub_component_type` 用 `string`（**不是 proto enum**），便于扩展新 engine kind 不破坏 proto——comment 必须明确</li><li>`ComponentTarget.replicas` 是 `optional int32`——unset = 「对该 component 无意见」（v9 引入的语义）</li><li>`ComponentTarget.type` 在 `OverrideResult` 中有意义；在 `ScalingProposal` 中 ignored——双重用途，comment 写清</li><li>**`PredictionData.predicted_num_req` / `predicted_isl` / `predicted_osl` 必须为 `optional float`**（P0-1 review 决议）。理由：PR 4 chain-augment partial-merge **强依赖** `HasField()` 区分「我说 0.0」vs「我没意见，保留 prev」；plain `float` 在 proto3 默认值是 0.0，会让 user-llm-predictor 输出 `(num_req=1200)` 时 isl/osl 字段被误判为「主动设 0」覆盖 baseline。**这是 PR 4 的 hard contract，PR 1 必须先满足**</li></ol> |
| 测试 | 无 |
| 依赖 | 1-1（不依赖 1-2，可与 1-2 并行）|
| 估算 | 0.5 天 |

---

### 1-4：4 个 stage service + 关联 messages

| 项 | 内容 |
|---|---|
| 实现位置 | `proto/dynamo/planner/plugin/v1/plugin.proto` |
| 范围 | 4 service（4 RPC）+ 8 message：<br/>**Services**：`PredictPlugin { Predict }`、`ProposePlugin { Propose }`、`ReconcilePlugin { Reconcile }`、`ConstrainPlugin { Constrain }`<br/>**Stage messages**：`PredictStageRequest`/`PredictStageResponse`、`ProposeStageRequest`/`ProposeStageResponse`、`ReconcileStageRequest`/`ReconcileStageResponse`、`ConstrainStageRequest`/`ConstrainStageResponse`、`ProposeResult`（reconcile 收前一阶段全集时用） |
| 字段细节来源 | 主文档 v10 line 1033-1167 |
| **关键 review point** | <ol><li>**`ProposeStageResponse.final` / `ReconcileStageResponse.final`** 的 comment 必须**逐字 copy** 主文档「完全覆盖」semantics——这是 v6 的核心语义变更，proto comment 是开发者唯一可见的规范</li><li>**`ConstrainStageResponse.final` 的 comment 必须明确「silently ignored」**——主文档 v10 line 1159-1166；proto 不能强制语义，靠 server 侧实现 + comment 双重保障</li><li>**`ConstrainStageResponse.override` SET 拒收**——主文档 v9 line 1142-1148 说明在 Register 阶段静态拒，runtime drop + audit；proto comment 必须保留</li><li>`ReconcileStageRequest.proposals` 字段 (`repeated ProposeResult`)：reconcile plugin 看得到所有 propose 输出，且每个带 `priority`——这是用户 reconcile plugin 拿来 reweight / filter 的依据</li><li>所有 stage response 的 `oneof result { accept / override / reject }` 三选一——proto3 oneof 默认空 = 视为 ACCEPT（主文档 line 814 流水线 skip behavior）</li></ol> |
| 测试 | 无 |
| 依赖 | 1-3（依赖 PipelineContext / ComponentTarget / OverrideResult） |
| 估算 | 0.5 天 |

---

### 1-5：PluginLifecycle service

| 项 | 内容 |
|---|---|
| 实现位置 | `proto/dynamo/planner/plugin/v1/plugin.proto` |
| 范围 | 1 service（**2 RPC**）+ 4 message：<br/>**Service**：`PluginLifecycle { Bootstrap / Reset }`<br/>**Messages**：`BootstrapRequest`、`BootstrapResponse`、`ResetRequest`、`ResetResponse` |
| 字段细节来源 | 主文档 v10 line 400-413 |
| **⚠ 注意** | **不**包含 `Snapshot` / `Restore` ——v10 已 YAGNI 删除。Implementation Breakdown line 83 仍写 `Bootstrap / Reset / Snapshot / Restore` 是过时信息，consistency check sub-task 会同步修正 |
| 接口约定（comment 内必须明确）| <ol><li>`Bootstrap`：plugin 启动后 **第一次** 被 orchestrator 调用，传入 plugin 启动需要的 `bootstrap_data`（如 benchmark FPM 字节流）；plugin 借此完成一次性初始化（如 prime regression model）</li><li>`Reset`：配置 reload 或 test setup/teardown 时调；plugin 内部 state 全部清空回到 Bootstrap 之前的状态</li><li>**不强制 idempotent**：Bootstrap 多次调用 plugin 可选自行去重，**但 orchestrator 保证只在生命周期开始调用一次**——这是双方契约</li><li>**Backward-compatible 演化承诺**：未来如果加 `Snapshot` / `Restore`，proto3 加 RPC 是 client 兼容的（旧 client 只是不调新 RPC）；不需要 v2 package</li></ol> |
| `BootstrapRequest` 字段（最小集）| `bytes bootstrap_data = 1;`（具体格式由 plugin 自己定，orchestrator 透传）<br/>`map<string, string> hints = 2;`（启动 hints，例如 "regression_kind: prefill"）<br/>注：v10 主文档没有定具体字段，本 sub-task 收紧到最小可用集；后续若有 plugin 需要更多元信息，加新 optional 字段即可 |
| 测试 | 无 |
| 依赖 | 1-1 |
| 估算 | 0.5 天 |

---

### 1-6：protoc 编译脚本 + CI 集成

| 项 | 内容 |
|---|---|
| 实现位置 | `tools/build/gen_planner_proto.sh`<br/>`components/src/dynamo/planner/plugins/proto_gen/`（生成产物） |
| 接口 | <pre>#!/usr/bin/env bash<br/># Generate Python stubs from proto/dynamo/planner/plugin/v1/plugin.proto<br/># into components/src/dynamo/planner/plugins/proto_gen/<br/>set -euo pipefail<br/>buf generate proto/dynamo/planner/plugin/v1/<br/># 或：python -m grpc_tools.protoc -Iproto --python_out=... --grpc_python_out=...<br/></pre> |
| 二选一决策 | **优先用 `buf`**（dynamo runtime 已有 buf.yaml；统一工具链）；若 dynamo 还没用 buf，fallback `grpc_tools.protoc` |
| 配置 | `buf.gen.yaml`（如果用 buf）：指定 `python` + `python-grpc` plugin |
| CI 集成 | <ol><li>在 `.github/workflows/pr.yaml` 现有 `planner-build` job 中加 `gen_planner_proto.sh` 调用（或者在 `pre-merge.yml` 的 lint 阶段）</li><li>**check-in 生成产物**：`proto_gen/*.py` 提交到 git；CI 跑 `gen_planner_proto.sh` 后 `git diff --exit-code` 验证生成产物与提交一致（防止开发者改 proto 忘 regen）</li></ol> |
| 测试 | `tools/build/test_gen_planner_proto.sh`：跑一次脚本，断言生成的 `*.py` 包含预期 service / message 名 |
| 依赖 | 1-2 / 1-3 / 1-4 / 1-5 全部完成（生成产物覆盖完整 schema） |
| 估算 | 0.5-1 天（取决于 buf vs grpc_tools 决策、CI 集成复杂度）|

---

### 1-7：Pydantic 镜像 (`types.py`)

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/plugins/types.py` |
| 范围 | 与 1-2 / 1-3 / 1-4 / 1-5 中**所有 message + enum** 一一对应的 Pydantic v2 class（**不**包含 service stub） |
| 为什么需要 Pydantic 镜像 | <ol><li>**in-process plugin 路径**（`InProcessTransport`）直接 Python 调用，proto generated class 用着别扭（builder pattern、wrapped optional 等）；Pydantic class 给 in-process 路径**零开销**且 Pythonic</li><li>**测试构造方便**：`OverrideResult(targets=[ComponentTarget(...)])` 比 `proto.OverrideResult(targets=[proto.ComponentTarget(...)])` 干净</li><li>**JSON 序列化** 为 audit log / debug：Pydantic `.model_dump()` 直接给 JSON</li></ol> |
| 接口示例 | <pre>class OverrideType(IntEnum):<br/>    SET = 0; AT_LEAST = 1; AT_MOST = 2<br/><br/>class ComponentTarget(BaseModel):<br/>    sub_component_type: str<br/>    component_name: str \| None = None<br/>    replicas: int \| None = None<br/>    type: OverrideType = OverrideType.SET<br/><br/>class OverrideResult(BaseModel):<br/>    targets: list[ComponentTarget]<br/>    reason: str = ""<br/><br/>class PipelineContext(BaseModel):<br/>    request_id: str<br/>    decision_id: str = ""<br/>    observations: ObservationData \| None = None<br/>    predictions:  PredictionData  \| None = None<br/>    proposal:     ScalingProposal \| None = None<br/>    constrained:  ScalingProposal \| None = None<br/>    ...</pre> |
| **字段对齐契约** | Pydantic class 字段名 / 类型 / optional-ness **必须**与 proto 一一对应（含字段顺序、enum 数值）；1-8 round-trip 测试就是验证这条契约 |
| 测试 | 留给 1-8 |
| 依赖 | 1-2 / 1-3 / 1-4 / 1-5 全部完成 |
| 估算 | 0.5 天 |

---

### 1-8：proto ↔ Pydantic round-trip 测试

| 项 | 内容 |
|---|---|
| 实现位置 | `components/src/dynamo/planner/tests/plugins/proto/test_round_trip.py` |
| 范围 | 对 1-7 中**所有 Pydantic class**（约 20+）写 round-trip 测试 |
| 接口 | <pre>@pytest.mark.parametrize("cls,proto_cls,sample", PROTO_PYDANTIC_PAIRS)<br/>def test_round_trip(cls, proto_cls, sample):<br/>    """pydantic -> proto -> pydantic 等价"""<br/>    pyd = cls(**sample)<br/>    pb = pydantic_to_proto(pyd, proto_cls)<br/>    pyd2 = proto_to_pydantic(pb, cls)<br/>    assert pyd == pyd2<br/><br/>@pytest.mark.parametrize(...)<br/>def test_round_trip_reverse(cls, proto_cls, sample):<br/>    """proto -> pydantic -> proto 等价（确认 wire 字节一致）"""<br/>    pb = build_proto(proto_cls, **sample)<br/>    pyd = proto_to_pydantic(pb, cls)<br/>    pb2 = pydantic_to_proto(pyd, proto_cls)<br/>    assert pb.SerializeToString() == pb2.SerializeToString()</pre> |
| 转换工具 | `pydantic_to_proto(pyd, proto_cls)` / `proto_to_pydantic(pb, cls)` —— 通过 `pyd.model_dump()` + `proto_cls(**dict)` + `MessageToDict` 实现；放在 `plugins/types.py` 旁边的 `_proto_bridge.py` |
| **PR 1 测试矩阵** | <ol><li>`PluginInfo` 全字段 + 部分 optional 缺失</li><li>`RegisterRequest` 所有 12 字段一次性</li><li>`PipelineContext` 含 `observations` filled / unfilled 各一</li><li>`OverrideResult` 多 target / 0 target 各一</li><li>`ProposeStageResponse` 三种 oneof 各一（accept / override / reject）+ `final=true/false`</li><li>`PluginLifecycle.BootstrapRequest` 含 `bootstrap_data` bytes</li></ol> |
| pytest markers | `pytest.mark.{pre_merge, planner, gpu_0, unit}`（自动接现有 `planner-test` job）|
| 依赖 | 1-6（需要 generated proto stub）+ 1-7 |
| 估算 | 1 天 |

---

### 1-9：README + 演化策略文档

| 项 | 内容 |
|---|---|
| 实现位置 | `proto/dynamo/planner/plugin/v1/README.md`（在 1-1 骨架基础上扩展） |
| 内容 | <ol><li>schema 概览：所有 service / message / enum 一表</li><li>演化策略（已在 1-1 proto 文件头列出）</li><li>FPM encoding 跨语言 dispatch 表（msgspec / proto / json）</li><li>新增 plugin 类型时如何加 service（步骤 1: 加 service / 步骤 2: 加 stage request/response / 步骤 3: regen / 步骤 4: 改 LocalPlannerOrchestrator pipeline）</li><li>**指引到主 DEP 文档**：详细语义不在 README 重复，只列出锚点链接</li></ol> |
| 依赖 | 1-2 / 1-3 / 1-4 / 1-5 全部完成（README 描述完整 schema）|
| 估算 | 0.5 天 |

---

## PR 1 总估算

- **单人**：~3.25 天（1-1: 0.25 + 1-2: 0.5 + 1-3: 0.5 + 1-4: 0.5 + 1-5: 0.5 + 1-6: 0.75 + 1-7: 0.5 + 1-8: 1.0 + 1-9: 0.5 ≈ 5 天 ——保守 4-5 天）
- **双人并行**：1.5-2 天
  - 工程师 A：1-1 → 1-2 → 1-4（registry/stage 链路）
  - 工程师 B：1-3 → 1-5（context/lifecycle）
  - 汇合：1-6 → 1-7 → 1-8 → 1-9 串行

注：原 Implementation Breakdown 估算 1-2 工程师 × 2-3 天偏乐观，因为没拆 Pydantic 镜像 + round-trip test 这一段（约 1.5 天）。本 PR 1 详细文档建议按 **单人 5 天 / 双人 2.5 天** 排期。

---

## PR 1 Acceptance Criteria

- [ ] `proto/dynamo/planner/plugin/v1/plugin.proto` 编译通过（`buf lint` + `buf generate`）
- [ ] generated `proto_gen/*.py` check-in 到 git；CI 跑 `gen_planner_proto.sh` 后 `git diff --exit-code` 通过（防 regen 漂移）
- [ ] `plugins/types.py` 全部 Pydantic class 与 proto 字段一一对应
- [ ] `tests/plugins/proto/test_round_trip.py` 全部通过（双向 round-trip + wire-bytes 等价）
- [ ] `proto/dynamo/planner/plugin/v1/README.md` 完整覆盖 9 service + 25+ message + 3 enum 概览
- [ ] CI `planner-test` job 自动包含 round-trip 测试（marker 自动 discovery）
- [ ] PR description 明确：本 PR **零 production 影响**（无业务逻辑、无新代码路径调用）
- [ ] PR description 标注：**所有后续 PR 必须 import `dynamo.planner.plugins.proto_gen` 或 `plugins.types`，不允许重复定义**

---

## 跨 Sub-task 必须协调的点

### 1. proto 字段顺序 = wire format 锁定

一旦 PR 1 ship，proto 字段 tag（`= N`）**永久不可改**。任何 reviewer 看到 PR 内 proto diff 都要 double check：
- 删字段 → 必须留 `reserved N;`
- 改类型 → 禁止；起新字段 + 旧字段标 deprecated
- 改 `optional` → 改成 `required` 不允许；改成更宽松的没问题

### 2. Pydantic class 与 proto 必须 lock-step 演化

后续 PR 改 proto 必须**同 PR 改 Pydantic class** + round-trip 测试加新字段 case。**禁止**只改一边——这是隐性数据契约违约。

CI 上加一条 `meta-check`：grep proto 文件的 message 名集合，比对 `types.py` 的 class 名集合——不一致 fail（在 `tests/plugins/proto/test_class_coverage.py` 实现，1-8 sub-task 顺手做）。

### 3. `PluginLifecycle` 简化决策必须出现在 README

主文档 v10 已 YAGNI 删除 `Snapshot` / `Restore`，但开发者读 v9 历史文档可能看到 4 个 RPC。1-9 README 必须明确：

> **PR 1 v1 lifecycle = 2 RPC only**（Bootstrap / Reset）。Snapshot / Restore 不在 PR 1 范围；如果未来需要，proto3 加新 RPC 是 backward-compatible，单独 PR 引入即可。

### 4. FPM encoding `bytes` 字段的「真实 schema」由谁定？

`FpmData.prefill_engines` 是 `map<string, bytes>`——proto 完全不约束 bytes 内容。**真正的 schema 在三个不同地方**：
1. `msgspec` 编码：现有 `forward_pass_metrics.py` 中的 `FPM_VERSION` + msgspec class
2. `proto` 编码：未来在**单独的 `fpm_engine.proto`** 中定义（**不在 PR 1 范围**）
3. `json` 编码：等价于 `proto` JSON wire format

PR 1 的 `RegisterRequest.fpm_encoding` 只是**协商通道**，真实 schema 留给 PR 6 / 后续 plugin 实现按需选用。1-9 README 里要明确这个 boundary。

---

## 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| proto 字段定义错（如类型选 `int32` 但应该 `int64`） | **中** | 1-2/1-3/1-4 review 时，逐字段 cross-ref 主文档 v10；任何字段类型与主文档不一致**直接 reject** |
| `buf` 工具链 dynamo runtime 未引入 → fallback grpc_tools 复杂度上升 | **低** | 1-6 子任务先 ~30min 探查 dynamo `tools/build/` 下是否已有 buf；没有就用 grpc_tools，不在 PR 1 引入新工具链 |
| Pydantic v1 vs v2 兼容性（dynamo 现状）| **低** | 先 grep dynamo 当前 Pydantic 版本；统一 v2（1-7 default）；如发现 v1 还在用，**只在 plugins 子模块**升 v2 不破坏其他模块 |
| `PluginLifecycle` 的 `BootstrapRequest.bootstrap_data` 字段过宽（`bytes`）→ 后续 plugin 自由发挥导致 proto 失去契约价值 | **低-中** | 1-5 接受 `bytes` 是 v1 妥协；1-9 README 标记「v2 应用 oneof 约束 bootstrap_data 类型」作为 follow-up |
| Implementation Breakdown line 83 写 4 RPC 与 v10 主文档不一致 → 开发者写 PR 1 时按错的 spec 实现 | **中** | consistency check sub-task **必须在 PR 1 启动前** 把 Implementation Breakdown 同步成 2 RPC；本 PR 1 文档已显式提示 |

---

## 推荐 staffing

- **1 名 Backend 工程师**：负责 1-1 / 1-2 / 1-4 / 1-6 / 1-9
- **1 名 Backend / Python 工程师**（兼）：负责 1-3 / 1-5 / 1-7 / 1-8
- **1 名 reviewer**：proto schema 审查（重点 1-2 / 1-3 / 1-4 字段细节、tag 分配、optional 与否）

---

## Resolved Questions（已决议）

### Q1：用 `buf` 还是 `grpc_tools.protoc`？

**决议**：1-6 子任务**先探查 dynamo 现状**——如果 `tools/build/` 或仓库根有 `buf.yaml`，直接复用；否则 fallback `grpc_tools.protoc`，**不在 PR 1 引入新工具链**。理由：PR 1 风险点应集中在 schema 定义，不应被工具链选择放大。

### Q2：proto generated `*.py` 是否 check-in？

**决议**：**check-in**。理由：
- dynamo 当前编译时机不统一（部分模块 build-time gen，部分 runtime import）；check-in 消除「import 时找不到 module」类问题
- CI 跑 `gen_planner_proto.sh` 后 `git diff --exit-code` 防止 proto / generated 漂移
- 缺点是 PR diff 看着大；但 generated 代码是 deterministic 的，review 时 reviewer 只看 `.proto` 即可

### Q3：是否在 PR 1 引入 `protoc-gen-validate` (PGV) 做字段校验？

**决议**：**不**。理由：
- PGV 引入额外依赖 + 学习成本，PR 1 风险已经在「定义正确性」上
- 校验逻辑在 server 侧（PluginRegistry / orchestrator）实现更合适——可以错误信息更友好
- v2 / 后续 PR 可以补 PGV，proto3 加 validation annotation 是 backward-compatible

### Q4：Pydantic 镜像放在 `plugins/types.py`（单文件）还是按 message 拆多文件？

**决议**：**单文件 `types.py`**（v1）。理由：
- 全部 message 加起来 ~25 个、~250 行，单文件 import 方便
- 后续如果增长（>500 行），1-7 sub-task 内**预留**按 stage 拆分的 fold 结构（用 `# region: PluginRegistry` 等 marker），重构成本低

### Q5：是否需要 `PluginLifecycle` 的 `Bootstrap` 在 v1 就拆出 fpm_specific 字段（如 `BootstrapFpmRequest`）？

**决议**：**不**。理由：
- v1 仅 `bytes bootstrap_data` 是有意为之的 generic 接口——builtin plugin 自己定 bytes 解释（如 builtin-throughput-propose 把 benchmark FPM 序列化进去）
- 如果未来发现「所有 builtin 都需要同样的 bootstrap shape」，再加新 message 类型不破坏现有契约
- YAGNI 原则与主文档 v10 删 Snapshot/Restore 一致

---

## 已删除的内容（v0 → v1）

无（首版）

---

## 与其他 PR 的接口

| 下游 PR | 依赖 PR 1 的内容 |
|---|---|
| PR 2（Transport + Clock） | `PluginRegistry.Register` 用作 transport contract 测试的 echo plugin；依赖 `PipelineContext` 做 in-process echo |
| PR 3（Registry + Scheduler） | 完整实现 `PluginRegistry` 4 RPC server + `PluginInfo` filling + `CircuitState` 状态机 |
| PR 4（Merge） | 依赖 `OverrideResult` / `ComponentTarget` / `OverrideType` / `RejectResult` 实现 type-aware merge |
| PR 5（Orchestrator） | 依赖 `PipelineContext` / 4 stage service / `ScalingProposal` 实现流水线驱动；依赖 `PluginLifecycle` 调 builtin plugin Bootstrap |
| PR 6（5 个 builtin） | 依赖 4 stage service signature + `PluginLifecycle` 实现真实 builtin |
| PR 7（NativePlannerBase 双路径） | 依赖 `ScalingProposal` → `TargetReplica` 转换 |
| PR 8（可观测性 / Replay） | 依赖 `PluginInfo` / `CircuitState` 上报 metrics |

**任何 PR 改 proto** → **必须**回 PR 1 模式：版本号 `protocol_version` bump + `RegisterResponse.negotiated_protocol_version` 协商 + 所有 PluginRegistry 客户端兼容旧版。

---

## 下一步

1. **review 本 PR 1 详细文档**（重点：sub-task 拆分、字段决策、Q1-Q5 决议）
2. **同步 Implementation Breakdown**（line 83 改 PluginLifecycle 为 2 RPC）—— consistency check sub-task 一起做
3. **review 后启动 PR 1 实施**（按 1-1 → 1-2 → ... → 1-9 顺序，或按 staffing 表并行）
