---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 请求迁移
---

<p align="left">
  <a href="./request-migration.md" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

本文档介绍 Dynamo 如何实现请求迁移，以便在 LLM 文本生成期间优雅地处理 worker 故障。请求迁移允许正在处理的请求在原始 worker 不可用时继续在其他 worker 上执行，从而提供故障容错能力并改善用户体验。

## 概述

请求迁移通过一个 Migration operator 实现，该 operator 位于 Backend operator 和服务后端之间的 LLM 处理流水线中。当 worker 在请求处理期间发生故障时，迁移系统会保留部分生成状态，并在新的 worker 上重新创建请求，从上一个 worker 中断的位置继续执行。

## 架构组件

### Migrator

迁移系统集成在前端预处理和实际服务后端之间的 LLM 处理流水线中。这个位置使它能够拦截所有通信流，并透明地管理故障场景。

主要职责：
- 拦截流经流水线的所有请求和响应
- 通过错误模式匹配检测 worker 故障场景
- 使用可配置的迁移限制管理重试逻辑
- 跟踪部分响应状态，以实现无缝续接

### 迁移限制配置

迁移限制在 **frontend** 级别配置，并全局应用于该 frontend 服务的所有模型。此参数指定一个请求最多可以迁移到另一个 worker 的次数：

- 默认行为：不允许迁移（migration_limit=0）
- 通过 frontend 上的 `--migration-limit` 标志设置
- 应用于该 frontend 服务的所有模型

### 最大序列长度配置

最大序列长度设置控制迁移系统为某个请求缓存 token 状态的时长。一旦总序列长度（prompt + 已生成 token）超过此限制，该请求的迁移就会被禁用，并停止 token 跟踪：

- 默认行为：无限制（未设置 `--migration-max-seq-len`）
- 通过 frontend 上的 `--migration-max-seq-len` 标志或 `DYN_MIGRATION_MAX_SEQ_LEN` 环境变量设置
- 防止因缓存长序列而导致内存无限增长
- 边界：恰好达到限制时仍可迁移；只有严格超过限制才会禁用迁移
- 检查会在请求初始化时（prompt 长度）和生成期间（prompt + 输出 token）运行

## Token 状态跟踪和请求迁移

迁移系统的核心能力是通过 token 状态管理来保留并继续部分生成。这确保当 worker 在生成中途发生故障时，新的 worker 可以从确切的故障点无缝继续。

### Token 累积过程

当请求正在处理且响应从 worker 回流时，迁移系统会跟踪每一个已成功生成的 token：

1. **初始请求状态**：系统从包含初始 prompt token 的原始预处理请求开始。

2. **响应跟踪**：当每个响应从 worker 到达时，迁移系统会提取新生成的 token，并将其追加到请求的 token 序列中。这样会累积所有已经生成的 token。

3. **Token 计数管理**：系统还会更新剩余 token 预算，以反映已经生成的 token 数量，确保总生成量保持在最初请求的限制范围内。

### 迁移触发场景

迁移系统处理两种不同的故障场景：

#### 1. 新请求迁移（初始连接失败）

**场景**：创建初始连接时 worker 不可达。

**错误模式**：通信系统报告所选 worker 实例不可用。

**迁移过程**：
- 在初始 stream 设置期间检测连接失败
- 递减迁移重试计数
- 尝试使用原始请求创建新的 stream
- 由于生成尚未开始，因此没有需要保留的部分状态

#### 2. 进行中请求迁移（stream 中途断开）

**场景**：在收到部分响应后的活跃生成期间连接丢失。

**错误模式**：在生成完成前检测到 stream 终止。

**迁移过程**：

1. **故障检测**：系统通过错误监控检测到 stream 断开。

2. **状态保留**：此时，请求的 token 序列同时包含原始 prompt token 和来自故障 worker 的所有已成功生成 token。

3. **创建新 Stream**：使用累积的请求状态创建新的 stream，确保新 worker 拥有完整上下文。

4. **继续生成**：新 worker 接收带有完整 token 上下文的请求，并从上一个 worker 中断的确切位置继续生成。

### 无缝 Token 流和请求状态演进

从客户端视角看，token 流是连续且不中断的。客户端会持续从第一个 worker 接收 token，直到发生故障，然后无缝地继续从备用 worker 接收 token，而不会感知到底层发生了迁移。

请求状态会在处理过程中动态演进。最初，请求只包含原始 prompt token。随着生成推进，每个成功生成的 token 都会追加到请求的 token 序列中，从而形成一条不断增长的完整对话上下文记录。

当迁移发生时，这个累积状态会传输给新的 worker，新的 worker 使用它重建完整上下文。随后，新 worker 会像从一开始就在处理该请求一样继续生成，但从序列中的当前位置开始。

迁移是透明的，因为：
1. 转换期间不会丢失或重复 token
2. 新 worker 通过累积的 token 序列获得完整上下文
3. 生成会从确切的故障点继续
4. 响应流保持一致的格式和时序

这种 token 累积机制确保迁移真正无缝，保留所有计算工作，并在 worker 切换过程中维持生成质量。

## 优点

1. **故障容错**：系统在单个 worker 故障期间仍能继续运行
2. **资源效率**：保留部分生成结果，而不是从头重新开始
3. **无缝用户体验**：用户在 worker 故障期间不会感到中断
4. **可配置行为**：迁移限制允许根据部署需求进行调优
5. **无 Token 丢失**：在迁移过程中完整保留生成状态

## 设计考量

迁移系统的设计包含几个重要的架构考量：

**多模型支持**：由于一个 frontend 可能同时服务多个模型，迁移限制在 frontend 级别配置，并统一应用于所有模型，从而简化运维管理。

**状态管理**：系统不仅仔细跟踪 token 序列，还会跟踪剩余 token 预算、停止条件和采样参数等元数据，以确保状态完整保留。

**错误处理**：迁移系统会区分不同类型的故障，并为每种场景应用合适的恢复策略。

## 监控和指标

迁移系统公开 Prometheus 指标，用于监控迁移活动。这些指标可在 frontend 的 `/metrics` 端点上获取（默认端口 8000）：

- `dynamo_frontend_model_migration_total`：跟踪请求迁移总次数的计数器
  - 标签：
    - `model`：正在服务的模型名称
    - `migration_type`：可以是 `new_request`（初始连接失败）或 `ongoing_request`（stream 中途断开）
- `dynamo_frontend_model_migration_max_seq_len_exceeded_total`：跟踪因为序列长度超过已配置的 `--migration-max-seq-len` 而禁用迁移的次数的计数器
  - 标签：
    - `model`：正在服务的模型名称

**指标输出示例：**
```text
dynamo_frontend_model_migration_total{migration_type="ongoing_request",model="Qwen/Qwen3-0.6B"} 3
dynamo_frontend_model_migration_total{migration_type="new_request",model="Qwen/Qwen3-0.6B"} 1
dynamo_frontend_model_migration_max_seq_len_exceeded_total{model="Qwen/Qwen3-0.6B"} 2
```

这些指标可用于：
- 监控 worker 可靠性和故障模式
- 在迁移率过高、表明存在基础设施问题时发出告警
- 跟踪故障容错机制的有效性
- 监控 `--migration-max-seq-len` 达到限制的频率，这可能表示需要调整该限制

有关 Dynamo 指标的更多信息，请参阅[指标文档](../observability/metrics.md)。

## 已知限制

### 多个选择（`n > 1`）

对于请求使用 `n > 1` 要求生成多个选择的 OpenAI 兼容请求，**不支持**请求迁移。即使 `--migration-limit` 大于 0，Dynamo 也会为这些请求禁用迁移。

**原因：** 多选择生成会维护各个选择各自的输出状态。迁移部分完成的请求需要分别传输每个选择的已生成 token 状态、剩余 token 预算、完成状态和 decoder 状态。当前迁移路径只保留单一续接状态，因此重试交错的 `n > 1` 请求可能会重复或丢失特定选择的输出。

此限制不影响省略 `n` 或将其设置为 1 的普通单选择请求。

### Guided Decoding（结构化输出）

对于使用 guided decoding（结构化输出 / JSON schema）的请求，**不支持**请求迁移。当 worker 在 guided-decoding 请求期间发生 stream 中途故障时，错误会传播给客户端，而不是尝试迁移。

**原因：** 推理后端会为每个新请求重新初始化 guided-decoding 有限状态机（FSM），并且只在新生成的 token 上推进它，而不会在上下文/prompt token 上推进它。当部分完成的请求迁移到新的 worker 时，新 worker 会把已生成 token 作为上下文重放，但 FSM 会从 schema 根节点开始。这种 token 状态和 FSM 状态之间的不匹配会产生损坏的输出，通常表现为重复或嵌套的 JSON。

此限制同样适用于所有后端（vLLM、SGLang、TRT-LLM）。

**未来方向：** 支持 guided-decoding 请求的迁移将需要在 worker 之间序列化并恢复 FSM 状态，或者在新的 worker 上通过 FSM 重放先前的输出 token。这被跟踪为未来增强功能。

## 运维影响

请求迁移从根本上改变了系统处理故障的方式，从“快速失败”方法转向“优雅降级”模型。这种架构转变在为客户端保持相同外部 API 契约的同时，实现了更高可用性和更好的资源利用率。
