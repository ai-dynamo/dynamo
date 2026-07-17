---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: 简介
---

<p align="left">
  <a href="./about.md" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

# Dynamo 简介

Dynamo 是一个开源、高吞吐、低延迟的推理框架，专为在分布式环境中服务生成式 AI 工作负载而设计。本页概述 Dynamo 的设计原则、性能优势和生产级功能。

> [!TIP]
> 想马上开始？请参阅 [Quickstart](quickstart.zh-CN.mdx)，几分钟内即可安装并运行 Dynamo。

## 为什么选择 Dynamo？

推理引擎优化 GPU；Dynamo 优化围绕它们的整个系统。

- **在任意引擎之上的系统级优化** -- 推理引擎优化单 GPU 前向传递。Dynamo 增加分布式层：分离式服务、智能路由、跨内存层级的 KV 缓存管理，以及自动扩缩容。
- **可组合的性能提升技术** -- 分离式服务、KV 缓存感知路由和 KV 缓存卸载等技术各自都能提升性能；组合使用时会带来叠加收益。
- **引擎无关** -- 可与 vLLM、SGLang 和 TensorRT-LLM 配合使用。无需改变服务基础设施即可替换引擎。正在扩展对 Intel XPU 和 AMD 硬件的支持。
- **面向大规模生产就绪** -- Dynamo 覆盖完整部署生命周期：自动配置（AIConfigurator）、运行时自动扩缩容（Planner）、拓扑感知的 gang scheduling（Grove）、容错和可观测性。
- **模块化采用** -- 可以从一个组件开始（例如，仅在现有引擎之上使用 Router 实现 KV 感知路由）。按需采用更多组件。每个组件都可通过 pip 独立安装。

## 设计原则

### 面向 AI 推理的坚实基础

Dynamo 在推理引擎之上增加系统级优化。为了提供这些优化，Dynamo 采用操作系统式的方法，为调度、内存管理和数据传输奠定基础。这些基础让 Dynamo 能随着新的系统级性能技术出现而持续演进。

Dynamo 进行系统级设计的动机之一，是支持分离式服务：在不同设备上运行 prefill 和 decode，使两者能够独立扩展和并行化。分离式服务需要三项能力：(1) 调度，用于在不相互干扰的情况下分配 prefill 和 decode 阶段；(2) 内存管理，用于 KV 缓存卸载和载入；(3) 低延迟数据传输，用于在节点之间以及跨内存层级移动 KV 缓存。

![Dynamo 基础：调度、内存管理和数据传输](../assets/img/intro-foundations.svg)

Dynamo 的基础能力最初面向分离式服务，随后扩展到用于多模态的 EPD 分离，现在支持 diffusion、RL 和 agents 等工作负载。

### 模块化且深度集成的生态系统

Dynamo 旨在降低生产环境中替换现有技术栈的负担。它以 Rust crates 和 pip wheels 的形式提供模块化、独立组件。例如，Dynamo 用于调度（Dynamo）、内存管理（KV Block Manager）和数据传输（NIXL）的三项基础能力都可以独立安装：

```bash
pip install ai-dynamo
pip install kvbm
pip install nixl
```

> [!NOTE]
> 也提供包含所有依赖项的预构建容器。请参阅 [Release Artifacts](../reference/release-artifacts.mdx) 了解容器镜像。

Dynamo 生态系统包含以下额外的模块化组件，并会随着时间继续增长：

| 类别 | 产品 | 描述 |
| :--- | :--- | :--- |
| **调度** | Dynamo | 面向 GenAI 工作负载的推理服务 |
| **路由** | Router | 利用 KV 缓存命中率和 KV 缓存负载进行智能路由。将添加更多算法（例如 agentic routing） |
| **数据传输** | [NIXL](https://github.com/ai-dynamo/nixl) | GPU 与分层存储之间的点对点数据传输（G1：GPU，G2：CPU，G3：SSD，G4：远程） |
| **内存** | KVBM (KV Block Manager) | 使用可自定义的淘汰策略，跨内存层级（G1-G4）管理 KV 缓存 |
| **扩缩容 / 云** | Planner | 在给定 SLA 约束（TTFT 和 TPOT）下，实时自动调优 prefill 和 decode 的性能 |
| | [Grove](https://github.com/ai-dynamo/grove) | 支持 Kubernetes 多节点分离式服务所需的 gang scheduling 和拓扑感知 |
| | [Model Express](https://github.com/ai-dynamo/model-express) | 通过缓存模型权重并经由 NIXL 将其传输到其他 GPU，快速加载模型权重。未来也将用于容错 |
| **性能** | [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) | 基于模型、ISL/OSL、HW 等估算聚合式与分离式服务的性能。以前称为 LLMPet |
| | [AIPerf](https://github.com/ai-dynamo/aiperf) | 用 Python 编写、重新架构的 GenAI-Perf，具有最大可扩展性；支持分布式基准测试 |
| | AITune | 给定模型或 pipeline，搜索最适合部署的后端（例如 TensorRT、Torch.compile 等）（即将推出） |
| | Flex Tensor | 从主机内存向 GPU 流式传输权重，以便在内存容量有限的 GPU 上运行超大语言模型（即将推出） |

这些组件是模块化的，但设计上会作为统一家族协同工作。新组件也将遵循同样的设计原则。

### 促进供应商无关的生态系统

Dynamo ***不是为供应商锁定而设计的***。Dynamo 旨在赋能更广泛的 AI 生态系统，并提供开发者所需的功能，例如与第三方组件集成。

从一开始，Dynamo 就设计为支持所有 LLM 推理引擎（vLLM、SGLang 和 TensorRT-LLM）。未来计划支持更多引擎，以覆盖更多开发者用例。

**也支持非 NVIDIA 硬件**：Dynamo 正在与 Intel 和 AMD 等 HW 供应商合作，扩展硬件支持。

受支持生态系统组件的完整列表：

| **产品领域** | **受支持的生态系统组件** |
| :--- | :--- |
| 推理引擎 | SGLang、TensorRT-LLM、vLLM |
| Kubernetes | Inference gateway |
| 内存管理 | Dynamo KV Block Manager、[LMCache](../integrations/lmcache-integration.md)、[SGLang HiCache](../backends/sglang/sglang-hicache.md)、[FlexKV](../integrations/flexkv-integration.md) |
| 网络和存储 | Mooncake、DOCA NetIO、GDS、POSIX、S3、3FS（[通过 NIXL 支持](../design-docs/kvbm-design.md)） |
| Multi-HW | Intel XPU、AMD |

## 性能

Dynamo 通过组合三项核心技术实现先进的 LLM 性能：分离式服务、KV 缓存感知路由和 KV 缓存卸载。这些技术由 NIXL 支撑，NIXL 是一种低延迟数据传输层，可在节点之间无缝移动 KV 缓存。

- [KV cache-aware routing](../design-docs/router-design.md) 根据 worker 负载和现有缓存命中情况智能路由请求。通过复用预先计算的 KV 对，它绕过 prefill 计算，立即开始 decode 阶段。[Baseten](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/#how-baseten-uses-nvidia-dynamo) 应用 Dynamo KV 缓存感知路由后，在 Qwen3 Coder 480B A35B 上实现了 2 倍更快的 TTFT 和 1.6 倍吞吐量。

- [KV cache offloading](../design-docs/kvbm-design.md) 通过将 KV 缓存从 HBM 移动到主机内存、本地磁盘或远程存储等成本更低的存储层，扩展可用上下文窗口。复用预计算状态可改善 TTFT、降低总体拥有成本（TCO），并支持更长上下文处理。

- [分离式服务](../design-docs/disagg-serving.zh-CN.md) 在“设计原则”部分，我们介绍了分离式服务的概念。[InferenceX](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs) 展示了它的性能。借助分离式服务和大规模专家并行，DeepSeek V3 可以达到约 7 倍吞吐量/GPU。
此外，当这三项技术组合在一起时，会产生如下图所示的叠加收益。

![分离式服务、KV 缓存感知路由和 KV 缓存卸载的性能可组合性](../assets/img/intro-perf.svg)

- **分离式服务 + KV Cache-Aware Routing** -- KV 缓存感知路由同时针对计算（prefill）和内存（decode）进行负载均衡，从而同时优化延迟和吞吐量。
- **分离式服务 + KV Cache Offloading** -- KV 缓存卸载带来更快的 TTFT，并且可以减少 prefill worker 数量以降低 TCO。
- **KV Cache-Aware Routing + KV Cache Offloading** -- 卸载会增加总体可寻址缓存大小，提高 KV 缓存命中率，进而加速 TTFT。

> [!TIP]
> 准备尝试这些技术？请参阅 [Dynamo recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes)，了解组合分离式服务、路由和卸载的分步部署示例。

## 从配置到生产级部署

### 使用 AIConfigurator 在 30 秒内找到最佳配置

手动为分离式服务寻找最佳并行度可能需要数天的穷举配置扫描，而这一挑战在大规模环境中只会更加严峻。

Dynamo 的 [AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/) 通过在 30 秒内识别性能最佳的配置来解决这一问题，并清晰预测相较于标准聚合式服务的性能提升。该逻辑原生集成到 Kubernetes Custom Resource Definition (CRD) 和 Dynamo Graph Deployment Request (DGDR) 中，使用户能够使用自动生成的优化配置进行部署。

### 使用 Planner 根据 SLA 自动调整部署

一旦通过 AIConfigurator 或 DGDR 找到离线配置，开发者就可以将所需模型部署到生产环境。然而，生产流量在线上可能变化很大，离线确定的静态配置将无法充分处理流量峰值。

Dynamo 提供 [Planner](../design-docs/planner-design.zh-CN.md) 来规避这个问题。开发者只需用 TTFT 和 Time Per Output Token (TPOT) 设置 SLA。Planner 会检查在线流量，并自动决定如何扩展 prefill 和 decode worker，从而在维持指定 SLA 的同时有效处理流量峰值。

最近，Planner 已扩展到处理更复杂的场景，例如在相同 SLA 下 Input Sequence Length (ISL) 剧烈变化。请参阅 [Planner documentation](../components/planner/planner-guide.zh-CN.md) 了解更多详细信息。

### 使用 Grove 应用拓扑感知的层级式 Gang Scheduling

当 Planner 决定自动扩缩容时，开发者需要一种方式来有效、独立且分层地扩展 worker。尤其对于 prefill/decode 解耦，prefill 和 decode worker 需要独立扩展以满足指定 SLA，并且需要被调度到物理上彼此接近的位置以获得最佳性能。

Dynamo 提供 [Grove](https://github.com/ai-dynamo/grove)，它是一个 Kubernetes operator，提供单一声明式 API，可用于编排从简单单 pod 部署到复杂多节点解耦系统的任意 AI 推理工作负载。

Grove 支持：

- 层级式 gang scheduling
- 拓扑感知放置
- 多级水平自动扩缩容
- 显式启动顺序
- 使用可配置替换策略的滚动更新

这些功能对于在数据中心规模部署和扩展推理以获得最佳性能至关重要。

### 确保 LLM 的容错能力

Kubernetes 自带一些容错功能，但 LLM 部署需要专门的容错和韧性机制。Dynamo 在多个层面提供全面的容错机制，确保生产部署中的 LLM 推理可靠运行：

- **Router and Frontend** -- Dynamo 支持启动多个 frontend + router 副本，并通过共享 router 状态提升容错能力。
- **Request Migration** -- 当 worker 在请求处理过程中失败时，Dynamo 可以将进行中的请求迁移到健康 worker，同时保留部分生成状态，并维持面向客户端的无缝 token 流。
- **Request Cancellation** -- Dynamo 支持通过 AsyncEngineContext trait 取消进行中的请求，该 trait 提供优雅停止信号和通过请求链传播的层级取消。
- **Request Rejection (Load Shedding)** -- 当 worker 过载时，Dynamo 会基于 KV 缓存利用率和 prefill token 的可配置阈值，以 HTTP 503 响应拒绝新请求。

### 可观测性

Dynamo 提供内置指标、分布式追踪和日志，用于监控推理部署。请参阅 [Observability Guide](../observability/README.md) 了解设置详情。

## 接下来做什么？

探索以下资源以深入了解：

- [Recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) -- 组合分离式服务、路由和卸载
- [KV Cache-Aware Routing](../components/router/router-guide.md) -- 配置智能请求路由
- [KV Cache Offloading](../components/kvbm/kvbm-guide.md) -- 设置多层内存管理
- [Planner](../components/planner/planner-guide.zh-CN.md) -- 配置基于 SLA 的自动扩缩容
- [Kubernetes Deployment](../kubernetes/README.md) -- 使用 Grove 进行大规模部署
- [Overall Architecture](../design-docs/architecture.zh-CN.md) -- 完整技术设计
- [Support Matrix](../reference/support-matrix.md) -- 检查硬件和引擎兼容性

**延伸阅读：** [Dynamo Digest](../digest/index.mdx)。
