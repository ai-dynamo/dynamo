---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
---

Dynamo KV Router 通过评估不同 worker 上的计算成本来智能地路由请求。它同时考虑解码成本（来自活动 block）和预填充成本（来自新计算的 block），并利用 KV cache 重叠来尽量减少重复计算。优化 KV Router 对于在分布式推理部署中实现最大吞吐量和最低延迟至关重要。

## 快速开始

我们可以通过 Dynamo frontend 使用 KV Router：

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

对于 Kubernetes，请在 Frontend service 上设置 `DYN_ROUTER_MODE=kv`。对于事件驱动的 KV 状态，请使用 [Router Operations](/dynamo/dev/components/router/router-operations#additional-notes) 中描述的后端专用 flag，配置 backend worker 发布 KV cache 事件。仅当你希望使用近似的 cache 状态预测时，才使用 `--no-router-kv-events`。

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--router-mode kv` | `round-robin` | 启用感知 KV cache 的路由 |
| `--load-aware` | disabled | 使用 KV 活动负载路由，不使用 cache 复用信号；在 frontend 上隐含启用 `--router-mode kv` |
| `--router-kv-overlap-score-credit` | `1.0` | 设备本地 prefix 重叠的 credit 乘数，范围从 0.0 到 1.0 |
| `--router-prefill-load-scale` | `1.0` | 在加入 decode block 之前，对调整后的 prompt 侧 prefill 负载进行缩放 |
| `--router-kv-events` / `--no-router-kv-events` | `--router-kv-events` | 消费 worker KV 事件，或在没有事件时回退到近似路由 |
| `--router-queue-threshold` | `16.0` | 背压队列阈值；通过 `nvext.agent_hints.priority` 启用优先级调度 |
| `--router-queue-policy` | `fcfs` | 队列调度策略：`fcfs`（尾部 TTFT）、`wspt`（平均 TTFT）或 `lcfs`（仅用于比较的反向排序） |
| `--no-router-track-prefill-tokens` | disabled | 在 router 负载统计中忽略 prompt 侧 prefill token；适用于仅 decode 的路由路径 |

### 独立 Router

你也可以将 KV router 作为独立服务运行（不使用 Dynamo frontend）。更多详细信息请参阅 [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/)。

有关部署模式和快速开始步骤，请参阅 [Router Guide](/dynamo/dev/user-guides/kv-cache-aware-routing)。有关 CLI 参数和调优指南，请参阅 [Configuration and Tuning](/dynamo/dev/components/router/configuration-and-tuning)。有关 A/B 基准测试，请参阅 [KV Router A/B Benchmarking Guide](/dynamo/dev/benchmarks/kv-router-ab-testing)。

## 前提条件和限制

**要求：**
- **仅支持动态 endpoint**：KV router 要求使用 `model_input=ModelInput.Tokens` 调用 `register_model()`。你的 backend handler 会接收带有 `token_ids` 的预分词请求，而不是原始文本。
- Backend worker 必须使用 `model_input=ModelInput.Tokens` 调用 `register_model()`（请参阅 [Backend Guide](/dynamo/dev/backends/custom-backend/python-workers-lower-level)）
- 使用 KV routing 时请使用动态发现，以便 router 跟踪 worker 实例及其 KV cache 状态

**多模态支持：**
- **通过多模态 hash 进行图像路由**：在已文档化的 TRT-LLM 和 vLLM router 路径中受支持。
- **其他 backend 或模态组合**：在依赖多模态 hash routing 之前，请检查相应 backend 的多模态文档。

**限制：**
- KV routing 不支持静态 endpoint；请使用动态发现，以便 router 跟踪 worker 实例及其 KV cache 状态

对于不使用 KV routing 的基础模型注册，请在静态和动态 endpoint 中使用 `--router-mode round-robin`、`--router-mode random`、`--router-mode least-loaded` 或 `--router-mode device-aware-weighted`。

## 后续步骤

- **[Router Guide](/dynamo/dev/user-guides/kv-cache-aware-routing)**：部署模式、快速开始和页面地图
- **[Routing Concepts](/dynamo/dev/components/router/routing-concepts)**：成本模型和 worker 选择行为
- **[Configuration and Tuning](/dynamo/dev/components/router/configuration-and-tuning)**：Router flag、传输模式和指标
- **[分离式服务](/dynamo/dev/components/router/disaggregated-serving)**：Prefill 和 decode 路由设置
- **[Router Operations](/dynamo/dev/components/router/router-operations)**：副本、持久化和恢复
- **[Router Examples](/dynamo/dev/components/router/router-examples)**：Python API 用法、K8s 示例和自定义路由模式
- **[Router Testing](https://github.com/ai-dynamo/dynamo/blob/cb05744fed2ef05bbd4a046f5dc7b4dbd1b38e9e/docs/components/router/router-testing.md)**：从 Rust 单元测试到基于 fixture 的 replay 和完整进程 E2E 的测试层级
- **[Standalone Indexer](/dynamo/dev/components/router/standalone-indexer)**：将 KV indexer 作为单独服务运行，以便独立扩缩容
- **[Router Design](/dynamo/dev/design-docs/component-design/router-design)**：架构细节、算法和事件传输模式
