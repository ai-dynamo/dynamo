---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 本地安装
sidebar-title: 本地安装
description: 使用容器或 PyPI 在本地机器或 VM 上安装并运行 Dynamo
---

本指南介绍如何在配备一个或多个 GPU 的本地机器或 VM 上安装并运行 Dynamo。完成后，你将拥有一个可工作的 OpenAI 兼容端点，用于提供模型服务。

对于生产环境的多节点集群，请参阅 [Kubernetes 部署指南](/dynamo/dev/kubernetes-deployment/start-here/kubernetes-quickstart)。如需为开发从源码构建，请参阅[从源码构建](/dynamo/zh-CN/dev/getting-started/building-from-source)。

## 系统要求

| 要求 | 支持范围 |
|---|---|
| **GPU** | NVIDIA Ampere、Ada Lovelace、Hopper、Blackwell |
| **OS** | Ubuntu 22.04、Ubuntu 24.04 |
| **架构** | x86_64、ARM64（ARM64 需要 Ubuntu 24.04） |
| **CUDA** | 12.9+ 或 13.0+（B300/GB300 需要 CUDA 13） |
| **Python** | 3.10、3.12 |
| **驱动** | 575.51.03+（CUDA 12）或 580.00.03+（CUDA 13） |

TensorRT-LLM 不支持 Python 3.11。

如需查看包含后端框架版本在内的完整兼容性矩阵，请参阅[支持矩阵](/dynamo/dev/resources/support-matrix)。

## 安装 Dynamo

### 选项 A：容器（推荐）

容器已预安装所有依赖项。无需额外设置。

```bash
# SGLang
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1

# TensorRT-LLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1

# vLLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
```

如需在同一个容器中运行 frontend 和 worker，可选择：

- 使用 `&` 在后台运行进程（请参阅下方“运行 Dynamo”部分），或
- 打开第二个终端并使用 `docker exec -it <container_id> bash`

如需查看可用版本，请参阅[发布产物](/dynamo/dev/resources/release-artifacts#container-images)；
如需运行说明，请参阅各后端指南：[SGLang](/dynamo/dev/backends/sg-lang) |
[TensorRT-LLM](/dynamo/dev/backends/tensor-rt-llm) | [vLLM](/dynamo/dev/backends/v-llm)

### 选项 B：从 PyPI 安装

仅支持 vLLM 和 SGLang。TensorRT-LLM 请使用选项 A。

```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv venv
source venv/bin/activate
uv pip install pip
```

为你选择的后端安装系统依赖项和 Dynamo wheel：

**SGLang**

```bash
sudo apt install python3-dev
uv pip install --prerelease=allow "ai-dynamo[sglang]"
```

对于 CUDA 13（B300/GB300），推荐使用容器。详情请参阅
[SGLang 安装文档](https://docs.sglang.io/get_started/install.html)。

**vLLM**

```bash
sudo apt install python3-dev libxcb1
uv pip install --prerelease=allow "ai-dynamo[vllm]"
```

## 运行 Dynamo

### 发现后端

Dynamo 组件通过共享后端相互发现。可使用两个选项：

| 后端 | 何时使用 | 设置 |
|---|---|---|
| **File** | 单机、本地开发 | 无需设置 -- 将 `--discovery-backend file` 传递给所有组件。事件平面会自动默认使用 ZMQ（无需 NATS）。 |
| **etcd** | 多节点、生产环境 | 需要正在运行的 etcd 实例（如果未指定标志，则为默认值）。事件平面默认使用 NATS。 |

本指南使用 `--discovery-backend file`。如需设置 etcd，请参阅[服务发现](/dynamo/dev/kubernetes-deployment/advanced-platform/service-discovery)。

### 验证安装（可选）

验证 CLI 已安装并可调用：

```bash
python3 -m dynamo.frontend --help
```

如果你克隆了仓库，可以运行其他系统检查：

```bash
python3 dev/sanity_check.py
```

### 启动 Frontend

```bash
# Start the OpenAI compatible frontend (default port is 8000)
python3 -m dynamo.frontend --discovery-backend file
```

如需在单个终端中运行（在容器中很有用），追加 `> logfile.log 2>&1 &`
以在后台运行进程：

```bash
python3 -m dynamo.frontend --discovery-backend file > dynamo.frontend.log 2>&1 &
```

### 启动 Worker

在另一个终端中（如果使用后台模式，也可以在同一个终端中），为你选择的后端启动 worker：

**SGLang**

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

**TensorRT-LLM**

```bash
python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

在这种本地单机设置中，警告 `Cannot connect to ModelExpress server/transport error. Using direct download.`
是预期行为（没有正在运行的 ModelExpress server），可以安全忽略。在配置了 `MODEL_EXPRESS_URL` 的 Kubernetes 部署中，
此警告，或相关的 `Failed to resolve local model path after server download`，
表示已配置 ModelExpress，但它实际上没有提供缓存模型；
请参阅 [Kubernetes 中的模型缓存](/dynamo/dev/kubernetes-deployment/model-loading/model-caching#option-2-modelexpress-p2p-distribution)
了解正确配置。

**vLLM**

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### KV Events 配置

对于无需依赖项的本地开发，请禁用 KV event 发布（避免使用 NATS）：

- **vLLM：** 添加 `--kv-events-config '{"enable_kv_cache_events": false}'`
- **SGLang：** 无需标志（默认禁用 KV events）
- **TensorRT-LLM：** 无需标志（默认禁用 KV events）

所有后端默认都禁用 KV events。对于 vLLM 和 SGLang，仅当你想启用 KV event 发布时，才添加后端专用的 `--kv-events-config`。对于 TensorRT-LLM，请使用 `--publish-events-and-metrics` 启用事件发布。

## 测试你的部署

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```

## 故障排除

**CUDA/驱动版本不匹配**

运行 `nvidia-smi` 检查你的驱动版本。Dynamo 对 CUDA 12 需要驱动 575.51.03+，对 CUDA 13 需要驱动 580.00.03+。B300/GB300 GPU 需要 CUDA 13。完整要求请参阅[支持矩阵](/dynamo/dev/resources/support-matrix)。

**模型无法装入 GPU（OOM）**

默认模型 `Qwen/Qwen3-0.6B` 需要约 2GB GPU 内存。更大的模型需要更多 VRAM：

| 模型大小 | 近似 VRAM |
|---|---|
| 7B | 14-16 GB |
| 13B | 26-28 GB |
| 70B | 140+ GB（多 GPU） |

从小模型开始，并根据你的硬件逐步扩展。

**TensorRT-LLM 与 Python 3.11**

TensorRT-LLM 仅支持容器路径。请使用 `tensorrtllm-runtime`
容器（选项 A）。

**容器运行但未检测到 GPU**

确保你向 `docker run` 传递了 `--gpus all`。如果没有此标志，容器将无法访问 GPU：

```bash
# Correct
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1

# Wrong -- no GPU access
docker run --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1
```

**vLLM worker 启动失败：FlashInfer sampler 的 JIT 与 CUDA 13 wheels**

在 CUDA 13 安装环境下运行 vLLM worker 时，worker 可能在启动阶段因 FlashInfer JIT 错误而中止：

```text
RuntimeError: Engine core initialization failed.
...
cuda/std/__cccl/cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are incompatible"
```

为 CUDA 13 安装解析出的 CUDA wheels 可能存在版本偏差：`torch` 将运行时头文件锁定到 13.0，而 vLLM 的 `tilelang` 依赖会拉取 `nvidia-cuda-nvcc` 13.2。FlashInfer 使用 `nvcc` 针对这些头文件编译其 sampler 内核，版本不匹配会导致构建失败。此问题在上游 [flashinfer#3493](https://github.com/flashinfer-ai/flashinfer/issues/3493) 中追踪。

设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，让 vLLM 回退到原生 sampler：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## 后续步骤

- [后端指南](/dynamo/dev/backends/sg-lang) -- 后端特定配置和功能
- [分离式服务](/dynamo/dev/user-guides/disaggregated-serving) -- 独立扩展 prefill 和 decode
- [KV Cache 感知路由](/dynamo/dev/user-guides/kv-cache-aware-routing) -- 智能请求路由
- [Kubernetes 部署](/dynamo/dev/kubernetes-deployment/start-here/kubernetes-quickstart) -- 生产环境多节点部署
