# Planner Advisory Mode 测试报告

## 1. 测试概述

| 项目 | 内容 |
|------|------|
| 测试功能 | Planner Advisory Mode（观测建议模式） |
| 测试日期 | 2026-03-24 |
| 测试人员 | Bruce Luo |
| 代码分支 | `feature/planner-advisory-mode` (brluobt/dynamo) |
| 代码 Commit | `2b0980981` - feat: implement Planner Advisory Mode |
| 测试结果 | **全部通过** |

## 2. 测试环境

### 硬件

| 项目 | 规格 |
|------|------|
| 节点 | l20-6 |
| 操作系统 | Ubuntu 22.04.4 LTS |
| GPU | NVIDIA L20 x 8 |

### 软件

| 组件 | 版本/镜像 |
|------|----------|
| Dynamo Operator | `dynamo-operator:local` |
| vLLM Runtime | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.1` |
| Advisory Mode 镜像 | `localhost:5000/dynamo:advisory-mode`（基于 `dynamo:gp-test` + 5 个修改文件叠加） |
| 推理模型 | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| NATS | 2.10.21 |
| etcd | 3.5.18 |

### 集群部署

测试在两个独立的 K8s namespace 中进行：

| Namespace | 模式 | Backend | GPU | 用途 |
|-----------|------|---------|-----|------|
| `gp-test` | Hierarchical Planner (5 个 DGD) | Mocker | 无 | Advisory Mode 功能验证 |
| `dynamo` | 单 DGD Disagg | vLLM | L20 | Advisory Mode 真实 GPU 环境验证 |

## 3. 测试方案

### 镜像构建

基于 `dynamo:gp-test` 镜像，叠加 Advisory Mode 修改的 5 个 Python 文件：

| 文件 | 改动内容 |
|------|---------|
| `defaults.py` | 新增 `ScalingMode` 枚举（ACTIVE/ADVISORY/NOOP） |
| `planner_config.py` | 新增 `scaling_mode`、`advisory_max_step_size`、`advisory_anomaly_threshold`、`advisory_file_output` 配置字段，向后兼容 `no_operation` 映射 |
| `planner_core.py` | 新增 15 个 Prometheus 指标、Advisory Engine 核心方法（`_emit_advisory_metrics`、`_estimate_sla_with_replicas`、`_build_path_recommendation`）、`/advisory/status` HTTP 端点 |
| `disagg_planner.py` | 4 处 `no_operation` 替换为 `scaling_mode` 三路分支，添加 advisory 输出调用 |
| `agg_planner.py` | 3 处 `no_operation` 替换为 `scaling_mode` 三路分支 |

### 部署方式

通过 `kubectl patch dgd` 更新 Planner 的镜像和配置参数：
- 镜像切换为 `localhost:5000/dynamo:advisory-mode`
- 入口命令切换为 `python3 -m dynamo.planner`
- 参数格式切换为 `--config` JSON 格式
- 配置 `scaling_mode: "advisory"` 和 `metric_reporting_prometheus_port: 9090`

## 4. 测试用例与结果

### 测试一：Mocker 环境 Advisory Mode（gp-test namespace）

**目的**：验证 Advisory Mode 基本功能，包括 Prometheus 指标输出、HTTP 端点、结构化日志。

**配置**：
```json
{
  "environment": "global-planner",
  "backend": "mocker",
  "mode": "prefill",
  "scaling_mode": "advisory",
  "throughput_adjustment_interval": 30,
  "metric_reporting_prometheus_port": 9090
}
```

**结果**：

| 验证项 | 状态 | 说明 |
|--------|------|------|
| Pod 启动 | 通过 | Pod `gp-prefill-0-planner-75f5d46c79-btq42` 正常启动，Running 状态 |
| Throughput 循环 | 通过 | 每 30 秒执行一次 adjustment interval |
| Advisory 日志 | 通过 | 日志中出现 `[ADVISORY] Recommendation` |
| Prometheus 指标 | 通过 | 全部 15 个新指标正常输出（见下表） |
| HTTP 端点 | 通过 | `/advisory/status` 返回完整 JSON |
| 未执行 Scaling | 通过 | Advisory 模式未调用 `set_component_replicas()`，其他 Pool Planner 不受影响 |

**Prometheus 指标采样**（无流量状态下）：
```
dynamo_planner_advisory_recommended_p    = 1.0
dynamo_planner_advisory_recommended_d    = 0.0
dynamo_planner_advisory_current_p        = 0.0
dynamo_planner_advisory_current_d        = 0.0
dynamo_planner_advisory_delta_p          = 1.0
dynamo_planner_advisory_scaling_action   = 1.0  (scale_up)
dynamo_planner_advisory_action_reason    = 1.0  (throughput_prediction)
dynamo_planner_advisory_est_ttft         = 60.46ms
dynamo_planner_advisory_ttft_headroom    = 1939.54ms
dynamo_planner_advisory_scaleup_total    = 7.0
dynamo_planner_advisory_scaledown_total  = 0.0
dynamo_planner_advisory_hold_total       = 0.0
```

**HTTP 端点响应**：
```json
{
  "scaling_mode": "advisory",
  "last_update": "2026-03-24T15:05:17.241941Z",
  "current": {"prefill": 0, "decode": 0},
  "recommended": {"prefill": 1, "decode": 0},
  "delta": {"prefill": 1, "decode": 0},
  "action": "scale_up",
  "reason": "throughput",
  "sla_estimation": {
    "est_ttft_ms": 60.46,
    "ttft_headroom_ms": 1939.54
  }
}
```

---

### 测试二：真实 GPU 环境 Advisory Mode（dynamo namespace）

**目的**：验证 Advisory Mode 在真实 GPU 推理环境中的表现，包括与 vLLM worker 的兼容性、真实流量下的指标准确性。

**配置**：
```json
{
  "environment": "kubernetes",
  "backend": "vllm",
  "throughput_adjustment_interval": 60,
  "profile_results_dir": "/mnt/profiling_data",
  "ttft": 2000,
  "itl": 50,
  "no_correction": true,
  "load_predictor": "constant",
  "scaling_mode": "advisory",
  "metric_reporting_prometheus_port": 9090
}
```

**部署拓扑**：
```
Frontend (vllm-runtime:0.9.1)
  └── Planner (advisory-mode 镜像, scaling_mode=advisory)
  └── Prefill Worker x1 (vllm-runtime:0.9.1, L20 GPU)
  └── Decode Worker x1 (vllm-runtime:0.9.1, L20 GPU)
```

**测试流程**：

1. 部署 Advisory Mode Planner
2. 验证 Planner 启动和 DGD 验证通过
3. 发送 11 个推理请求（1 个单独 + 10 个并发）
4. 等待下一个 adjustment interval 观察 advisory 输出

**结果**：

| 验证项 | 状态 | 说明 |
|--------|------|------|
| Pod 启动 | 通过 | DGD 验证成功，检测到 GPU: prefill=1, decode=1 |
| 模型检测 | 通过 | 自动检测到 `nvidia/Llama-3.1-8B-Instruct-FP8` |
| Metrics Validity Gate | 通过 | 无流量时正确跳过 advisory 输出："Metrics contain None or NaN values" |
| 真实流量采集 | 通过 | 观测到 `num_req: 12.00, isl: 21.27, osl: 113.73` |
| 实测延迟 | 通过 | `TTFT: 229.75ms, ITL: 14.42ms`（均在 SLA 范围内） |
| Prefill 副本计算 | 通过 | `4.25(需求) / 1600.00(容量) = 1(需要 1P)` |
| Decode 副本计算 | 通过 | `22.75(需求) / 300.00(容量) = 1(需要 1D)` |
| Advisory 建议 | 通过 | 建议 hold（1P1D → 1P1D），决策合理 |
| SLA 估算 | 通过 | `est_ttft: 120.0ms, est_itl: 15.0ms`（基于 profiling data） |
| SLA 余量 | 通过 | `ttft_headroom: 1880.0ms, itl_headroom: 35.0ms`（大量余量） |
| 推理服务不受影响 | 通过 | GPU 推理正常响应，Advisory Mode 未执行任何实际 scaling |

**Prometheus 指标采样**（有流量状态下）：
```
dynamo_planner_advisory_recommended_p    = 1.0
dynamo_planner_advisory_recommended_d    = 1.0
dynamo_planner_advisory_current_p        = 1.0   (正确读取 DGD 实际副本数)
dynamo_planner_advisory_current_d        = 1.0
dynamo_planner_advisory_delta_p          = 0.0
dynamo_planner_advisory_delta_d          = 0.0
dynamo_planner_advisory_scaling_action   = 0.0   (hold)
dynamo_planner_advisory_action_reason    = 6.0   (no change needed)
dynamo_planner_advisory_est_ttft         = 120.0ms
dynamo_planner_advisory_est_itl          = 15.0ms
dynamo_planner_advisory_ttft_headroom    = 1880.0ms
dynamo_planner_advisory_itl_headroom     = 35.0ms
dynamo_planner_advisory_hold_total       = 1.0
```

**HTTP 端点响应**：
```json
{
  "scaling_mode": "advisory",
  "last_update": "2026-03-24T15:14:03.530000Z",
  "current": {"prefill": 1, "decode": 1},
  "recommended": {"prefill": 1, "decode": 1},
  "delta": {"prefill": 0, "decode": 0},
  "action": "hold",
  "reason": "throughput",
  "sla_estimation": {
    "est_ttft_ms": 120.0,
    "est_itl_ms": 15.0,
    "ttft_headroom_ms": 1880.0,
    "itl_headroom_ms": 35.0
  }
}
```

## 5. 安全防护验证

| 防护机制 | 状态 | 验证方式 |
|----------|------|---------|
| Metrics Validity Gate | 通过 | 无流量时日志显示 "Metrics contain None or NaN values, skipping adjustment"，advisory 指标未输出 |
| 未执行实际 Scaling | 通过 | Advisory 模式下 Worker 副本数始终不变，`kubectl get pods` 确认无 Pod 增减 |
| 向后兼容 | 通过 | 其他 Pool Planner（gp-prefill-1, gp-decode-0, gp-decode-1）完全不受影响，继续使用原镜像正常运行 |
| HTTP 端点安全 | 通过 | `/advisory/status` 只读端点，不暴露敏感信息 |

## 6. 已知限制

| 限制 | 说明 |
|------|------|
| `advisory_current_p/d` 在 prefill-only 模式下不完整 | `gp-test` 中 prefill pool 的 Planner 以 `mode=prefill` 运行，`advisory_current_d` 始终为 0（因为不查询 decode worker 状态）。这是预期行为，disagg 模式下完整可见。 |
| SLA 估算基于 profiling data | `advisory_est_ttft` 和 `advisory_est_itl` 依赖预部署 profiling 数据的准确性，实际值可能因运行时条件偏差 |
| JSONL 文件输出未测试 | 本次测试未启用 `advisory_file_output`（需要设置 `log_dir`），此功能待后续验证 |
| Grafana Dashboard 未测试 | Dashboard JSON 尚未创建，可视化效果待后续验证 |

## 7. 结论

Planner Advisory Mode 在 Mocker 环境和真实 GPU 环境中均通过测试，核心功能正常：

1. **三模式架构**正常工作：advisory 模式不执行 scaling，但完整输出决策建议
2. **15 个 Prometheus 指标**全部正确输出，数值合理
3. **HTTP 端点** `/advisory/status` 返回结构化 JSON，方便快速查看
4. **安全防护**生效：无效 metrics 时跳过输出，不影响现有推理服务
5. **向后兼容**：未修改的 Planner 继续使用原镜像正常运行

**建议下一步**：
- 在持续负载下长时间运行（>24h）验证稳定性
- 创建 Grafana Advisory Dashboard 验证可视化效果
- 测试高负载场景下的 scale_up 建议准确性
- 提交 PR 到 ai-dynamo/dynamo 上游仓库
