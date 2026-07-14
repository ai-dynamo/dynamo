---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
---

Dynamo KV Router 通过评估不同 worker 上的计算成本来智能地路由请求。它同时考虑解码成本（来自活动 block）和预填充成本（来自新计算的 block），并利用 KV cache 重叠来尽量减少重复计算。优化 KV Router 对于在分布式推理部署中实现最大吞吐量和最低延迟至关重要。

## 快速开始

事件驱动的 KV 路由需要同时配置 frontend 和参与 KV 路由的后端 worker：

1. 启动 frontend 并启用 KV router：

   ```bash
   python -m dynamo.frontend --router-mode kv --http-port 8000
   ```

2. 使用相应后端的配置启用 KV 事件发布。

   对于 vLLM，请将 `--kv-events-config` 传递给每个 aggregated worker 或 disaggregated prefill worker：

   ```bash
   python -m dynamo.vllm --model Qwen/Qwen3-0.6B \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
   ```

   对于 SGLang，请将 `--kv-events-config` 传递给每个 aggregated worker。在 disaggregated 部署中，请将其传递给 prefill 和 decode worker：

   ```bash
   python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
   ```

不要将 `--kv-events-config` 传递给 vLLM 的 decode-only worker。对于共享网络命名空间的 worker，请使用唯一的 ZMQ 端点端口。对于 TensorRT-LLM，请传递 `--publish-kv-events`。[vLLM](../../../../../../docs/backends/vllm/vllm-reference-guide.md#kv-event-publication-for-kv-routing) 和 [SGLang](../../../../../../docs/backends/sglang/sglang-reference-guide.md#kv-events) 参考指南记录了后端默认值、各 worker 角色的行为和调优方法。

对于 Kubernetes，请在 Frontend service 上设置 `DYN_ROUTER_MODE=kv`，并按照上述说明将后端事件参数添加到相应的 vLLM 或 SGLang worker。对于不使用 worker 事件的近似路由，请省略后端事件参数，并在 frontend 上设置 `--no-router-kv-events` 或 `DYN_ROUTER_USE_KV_EVENTS=false`。

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--router-mode kv` | `round-robin` | 启用感知 KV cache 的路由 |
| `--load-aware` | disabled | 使用 KV 活动负载路由，不使用 cache 复用信号；在 frontend 上隐含启用 `--router-mode kv` |
| `--router-kv-overlap-score-credit` | `1.0` | 设备本地 prefix 重叠的 credit 乘数，范围从 0.0 到 1.0 |
| `--router-prefill-load-scale` | `1.0` | 在加入 decode block 之前，对调整后的 prompt 侧 prefill 负载进行缩放 |
| `--router-kv-events` / `--no-router-kv-events` | `--router-kv-events` | 消费 worker KV 事件；使用 `--no-router-kv-events` 可显式切换到近似路由 |
| `--router-queue-threshold` | disabled | 背压队列阈值；设置数值后启用队列，`nvext.agent_hints.priority` 会对等待中的请求重新排序 |
| `--router-queue-policy` | `fcfs` | 队列调度策略：`fcfs`（尾部 TTFT）、`wspt`（平均 TTFT）或 `lcfs`（仅用于比较的反向排序） |
| `--no-router-track-prefill-tokens` | disabled | 在 router 负载统计中忽略 prompt 侧 prefill token；适用于仅 decode 的路由路径 |

> [!IMPORTANT]
> 在默认的 `--router-kv-events` 设置下，如果 worker 不发布事件，router 会保持事件驱动模式，但无法获得真实的 cache 状态；router 不会自动切换到近似预测。请按上文配置后端对应的事件发布参数。
> 如果 worker 不发布事件，请使用 `--no-router-kv-events` 启用近似 cache 预测，或使用 `--load-aware` 仅按负载路由。

### 独立 Router

你也可以将 KV router 作为独立服务运行（不使用 Dynamo frontend）。更多详细信息请参阅 [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/)。

有关部署模式和快速开始步骤，请参阅 [Router Guide](../../../../../../docs/components/router/router-guide.md)。有关 CLI 参数和调优指南，请参阅 [Configuration and Tuning](../../../../../../docs/components/router/router-configuration.md)。有关 A/B 基准测试，请参阅 [KV Router A/B Benchmarking Guide](../../../../../../docs/benchmarks/kv-router-ab-testing.md)。

## 前提条件和限制

**要求：**
- **仅支持动态 endpoint**：KV router 要求使用 `model_input=ModelInput.Tokens` 调用 `register_model()`。你的 backend handler 会接收带有 `token_ids` 的预分词请求，而不是原始文本。
- Backend worker 必须使用 `model_input=ModelInput.Tokens` 调用 `register_model()`（请参阅 [Backend Guide](../../../../../../docs/development/backend-guide.md)）
- 使用 KV routing 时请使用动态发现，以便 router 跟踪 worker 实例及其 KV cache 状态

**多模态支持：**
- **通过多模态 hash 进行图像路由**：在已文档化的 TRT-LLM 和 vLLM router 路径中受支持。
- **其他 backend 或模态组合**：在依赖多模态 hash routing 之前，请检查相应 backend 的多模态文档。

**限制：**
- KV routing 不支持静态 endpoint；请使用动态发现，以便 router 跟踪 worker 实例及其 KV cache 状态

对于不使用 KV routing 的基础模型注册，请在静态和动态 endpoint 中使用 `--router-mode round-robin`、`--router-mode random`、`--router-mode least-loaded` 或 `--router-mode device-aware-weighted`。

## 后续步骤

- **[Router Guide](../../../../../../docs/components/router/router-guide.md)**：部署模式、快速开始和页面地图
- **[Routing Concepts](../../../../../../docs/components/router/router-concepts.md)**：成本模型和 worker 选择行为
- **[Configuration and Tuning](../../../../../../docs/components/router/router-configuration.md)**：Router flag、传输模式和指标
- **[分离式服务](../../../../../../docs/components/router/router-disaggregated-serving.md)**：Prefill 和 decode 路由设置
- **[Router Operations](../../../../../../docs/components/router/router-operations.md)**：副本、持久化和恢复
- **[Router Examples](../../../../../../docs/components/router/router-examples.md)**：Python API 用法、K8s 示例和自定义路由模式
- **[Router Testing](../../../../../../docs/components/router/router-testing.md)**：从 Rust 单元测试到基于 fixture 的 replay 和完整进程 E2E 的测试层级
- **[Standalone Indexer](../../../../../../docs/components/router/standalone-indexer.md)**：将 KV indexer 作为单独服务运行，以便独立扩缩容
- **[Router Design](../../../../../../docs/design-docs/router-design.md)**：架构细节、算法和事件传输模式
