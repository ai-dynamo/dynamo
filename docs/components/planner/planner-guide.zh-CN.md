---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Guide
---

<p align="left">
  <a href="./planner-guide.md" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

Dynamo Planner 是一个自动扩缩容控制器，会在运行时调整 prefill 和 decode 引擎的副本数，以满足延迟 SLA。它读取流量信号（Prometheus 指标或负载预测器输出）和引擎性能画像，用于决定何时扩容或缩容。

如需快速概览，请参阅 [Planner overview](README.md)。如需了解架构内部机制，请参阅 [Planner Design](../../design-docs/planner-design.md)。

## 扩缩容模式

planner 支持三个优化目标，这些目标决定扩缩容决策的方式：

- **`throughput`**（默认）：基于队列深度和 KV cache 利用率使用静态阈值。不需要 SLA 目标或 profiling。开箱即用。
- **`latency`**：与 `throughput` 采用相同方法，但使用更激进的阈值，即更早扩容并容忍更少排队。适合对延迟敏感的工作负载。
- **`sla`**：使用基于回归的性能模型，并指定 TTFT/ITL 目标。支持基于吞吐量（预测式）和基于负载（反应式）的扩缩容模式。适合需要精确 SLA 控制的高级用户。

**何时使用哪种模式：**

- 从 **`throughput`**（默认）开始，它无需配置即可立即工作。
- 如果工作负载有严格的延迟要求，并且你更倾向于过度预配置而不是排队，请切换到 **`latency`**。
- 当你有部署前 profiling 数据，并希望以特定 TTFT/ITL 值为目标时，请使用 **`sla`**。

## PlannerConfig 参考

planner 通过 `PlannerConfig` JSON/YAML 对象进行配置。使用 profiler 时，该对象位于 DGDR 规范的 `features.planner` 部分下：

```yaml
features:
  planner:
    mode: disagg
    backend: vllm
    # optimization_target defaults to "throughput" — works out of the box
```

对于基于 SLA 的扩缩容：

```yaml
features:
  planner:
    optimization_target: sla
    enable_throughput_scaling: true
    enable_load_scaling: false
    pre_deployment_sweeping_mode: rapid
    mode: disagg
    backend: vllm
```

若要在不改变副本数的情况下评估 Planner 行为，请启用 advisory 模式：

```yaml
features:
  planner:
    advisory: true
```

advisory 模式仅提供建议。Planner 会计算建议副本数、记录日志、将其导出为诊断信息，并显示在 HTML 报告中。这些建议不会作为扩缩容决策应用：Planner 不会执行扩缩容操作，也不会更改部署。

### 优化目标

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `optimization_target` | string | `throughput` | `throughput`：基于队列/利用率阈值扩缩容。`latency`：激进的低延迟阈值。`sla`：基于回归的扩缩容，并使用 ttft_ms/itl_ms 目标。 |

当 `optimization_target` 为 `throughput` 或 `latency` 时，会自动启用基于负载的扩缩容，并禁用基于吞吐量的扩缩容。`ttft_ms`/`itl_ms` 字段会被忽略。

### 扩缩容模式字段（SLA 模式）

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `enable_throughput_scaling` | bool | `true` | 启用基于吞吐量的扩缩容（需要部署前性能数据）。仅在 `optimization_target: sla` 时使用。 |
| `enable_load_scaling` | bool | `false` | 启用基于负载的扩缩容。仅在 `optimization_target: sla` 时使用。 |

使用 `optimization_target: sla` 时，必须至少启用一种扩缩容模式。

### 部署前扫描

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `pre_deployment_sweeping_mode` | string | `rapid` | 如何生成引擎性能数据：`rapid`（AIC 仿真，约 30 秒）、`thorough`（真实 GPU，2-4 小时）或 `none`（跳过）。 |

启用基于吞吐量的扩缩容时，planner 需要引擎性能数据。启动时，它会先尝试从 `get_perf_metrics` Dynamo 端点获取自基准测试结果（参见 PR #7779）。如果不可用，则回退到 `profile_results_dir` 中 profiler 生成的数据（npz 或 JSON）。这两种来源都会转换为 ForwardPassMetrics，并输入到 FPM 回归模型中。

### 基于吞吐量的扩缩容设置

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `throughput_adjustment_interval_seconds` | int | `180` | 基于吞吐量的扩缩容决策之间的秒数。 |
| `throughput_metrics_source` | string | `frontend` | 用于吞吐量扩缩容的 Prometheus 流量来源：`frontend` 从公共 Frontend 读取 `dynamo_frontend_*` 指标；`router` 从 LocalRouter 读取 `dynamo_component_router_*` 指标。在 GlobalPlanner 部署中，为池本地 Planner 使用 `router`。 |
| `min_endpoint` | int | `1` | 要维持的引擎端点最小数量。 |
| `max_gpu_budget` | int | `8` | planner 可以分配的 GPU 总数上限。 |
| `ttft_ms` | float | `500.0` | 用于扩缩容决策的 TTFT SLA 目标（毫秒）。 |
| `itl_ms` | float | `50.0` | 用于扩缩容决策的 ITL SLA 目标（毫秒）。 |

### 基于负载的扩缩容设置

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `load_adjustment_interval_seconds` | int | `5` | FPM 回归更新和基于负载的扩缩容决策之间的秒数。即使只启用了吞吐量扩缩容，实时 FPM 观测也会按此间隔输入到回归中。必须短于 `throughput_adjustment_interval_seconds`。 |
| `max_num_fpm_samples` | int | `64` | 为回归保留的 FPM 观测最大数量。 |
| `fpm_sample_bucket_size` | int | `16` | 用于观测淘汰的 bucket 数量（必须是完全平方数）。 |
| `load_scaling_down_sensitivity` | int | `80` | 缩容敏感度 0-100（0=永不，100=激进）。 |
| `load_metric_samples` | int | `10` | 每次决策要收集的指标样本数。 |
| `load_min_observations` | int | `5` | 做出扩缩容决策前所需的最少观测数。 |

### 通用设置

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `mode` | string | `disagg` | Planner 模式：`disagg`、`prefill`、`decode` 或 `agg`。 |
| `backend` | string | `vllm` | Backend：`vllm`、`sglang`、`trtllm` 或 `mocker`。 |
| `environment` | string | `kubernetes` | 运行时环境：`kubernetes`、`virtual` 或 `global-planner`。 |
| `namespace` | string | env `DYN_NAMESPACE` | 部署的 Kubernetes namespace。 |
| `advisory` | bool | `false` | 仅建议模式。计算、记录、导出并报告建议副本数，但不执行扩缩容操作，也不更改部署。 |

### 流量预测设置

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `load_predictor` | string | `arima` | 预测方法：`constant`、`arima`、`kalman` 或 `prophet`。 |
| `load_predictor_log1p` | bool | `false` | 在预测前对负载数据应用 log1p 变换。 |
| `prophet_window_size` | int | `50` | Prophet predictor 的窗口大小（秒）。 |
| `load_predictor_warmup_trace` | string | `null` | 用于引导预测的 warmup trace 文件路径。 |

### Kalman Filter 设置

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `kalman_q_level` | float | `1.0` | level 组件的 process noise。 |
| `kalman_q_trend` | float | `0.1` | trend 组件的 process noise。 |
| `kalman_r` | float | `10.0` | measurement noise。 |
| `kalman_min_points` | int | `5` | Kalman 预测激活前所需的最少数据点数。 |

### 诊断报告

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|-------------|
| `report_interval_hours` | float or `null` | `24.0` | 每 N 小时（模拟时间）生成一份 HTML 诊断报告。设置为 `null` 可禁用周期性报告生成。 |
| `report_output_dir` | string | `./planner_reports` | HTML 诊断报告的目录。 |
| `live_dashboard_port` | int | `8080` | 实时诊断 dashboard HTTP 服务器的端口。设置为 `0` 可禁用。启用后，访问 `http://host:port/` 查看累积快照的实时 Plotly 报告。 |

这些报告中展示的同一组诊断信号也会以 `dynamo_planner_*` 前缀导出为 Prometheus 指标，例如估算的 TTFT/ITL（`dynamo_planner_estimated_ttft_ms`、`dynamo_planner_estimated_itl_ms`）、建议副本数（`dynamo_planner_predicted_num_prefill_replicas`、`dynamo_planner_predicted_num_decode_replicas`）、每个引擎的容量和 FPM 队列深度，以及负载/吞吐量扩缩容决策枚举。

Replica Counts 图会将实际 prefill/decode 副本与 Planner 建议的 prefill/decode 副本的离散建议标记叠加显示。当 `advisory: true` 时，这些建议数量仅作为建议；Planner 会记录它本会执行的操作，但不会应用该变更。

## 与 Profiler 集成

当 profiler 在启用 planner 的情况下运行时，它会：

1. 选择最佳 prefill 和 decode 引擎配置
2. 生成引擎性能数据（prefill TTFT vs ISL、decode ITL vs KV-cache 利用率）
3. 将 `PlannerConfig` 和性能数据保存到独立的 Kubernetes ConfigMaps 中
4. 将 planner 服务添加到生成的 DGD，并配置为从这些 ConfigMaps 读取

planner 通过 `--config /path/to/planner_config.json` 接收其配置，该文件从 `planner-config-XXXX` ConfigMap 挂载。Profiling 数据从 `planner-profile-data-XXXX` ConfigMap 挂载。

请参阅 [Profiler Guide](../profiler/profiler-guide.md)，了解完整 profiling 工作流以及如何配置部署前扫描。

## 分层部署

如果你希望一个模型拥有一个公共端点，但同时有多个针对不同请求类别优化的私有 DGD，请使用分层部署：

- 一个包含 `Frontend`、`GlobalRouter` 和 `GlobalPlanner` 的 control DGD
- 一个或多个 prefill pool DGD
- 一个或多个 decode pool DGD

在当前工作流中，请为每个目标 pool 独立运行 profiling，然后手动组合最终的 control DGD 和 pool DGD。请参阅 [Global Planner Guide](global-planner.md)。

## 另请参阅

- [Planner overview](README.md) — 为什么 LLM 推理需要不同的 autoscaler
- [Planner Design](../../design-docs/planner-design.md) — 架构和算法内部机制
- [Planner Examples](planner-examples.md) — DGDR YAML 示例、样例配置、高级模式
- [Global Planner Guide](global-planner.md) — 多 DGD 协调、共享 GPU 预算、单端点多 pool 部署
- [Profiler Guide](../profiler/profiler-guide.md) — profiling 数据的生成方式
