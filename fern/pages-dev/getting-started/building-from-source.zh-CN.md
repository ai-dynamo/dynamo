---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 从源代码构建
sidebar-title: 从源代码构建
description: 从源代码构建 Dynamo，用于开发和贡献
---

<p align="left">
  <a href="./building-from-source.md" hreflang="en">English</a> | <strong>简体中文</strong>
</p>

当你想贡献代码、测试开发分支上的功能，或自定义构建时，可以从源代码构建 Dynamo。如果你只是想运行 Dynamo，[本地安装](local-installation.zh-CN.md)指南会更快。

本指南涵盖 Ubuntu 和 macOS。如需一个能自动处理所有这些步骤的容器化开发环境，请参阅 [DevContainer](#devcontainer)。

## 1. 安装系统库

**Ubuntu：**

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS：**

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake protobuf

# Verify Metal is accessible
xcrun -sdk macosx metal
```

## 2. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## 3. 创建 Python 虚拟环境

如果你还没有安装 [uv](https://docs.astral.sh/uv/#installation)，请先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建并激活虚拟环境：

```bash
uv venv .venv
source .venv/bin/activate
```

## 4. 安装构建工具

```bash
uv pip install pip 'maturin[patchelf]'
```

[Maturin](https://github.com/PyO3/maturin) 是 Rust-Python 绑定的构建工具。`patchelf` extra 让 maturin 能在构建期间修补原生扩展库路径。

## 5. 构建 Rust 绑定

```bash
cd lib/bindings/python
maturin develop --uv
```

## 6. 安装 GPU Memory Service

```bash
# Return to project root
cd "$(git rev-parse --show-toplevel)"
uv pip install -e lib/gpu_memory_service
```

## 7. 安装 Wheel

请安装带有后端附加依赖（backend extra）的 Dynamo，以拉取推理引擎及其 CUDA 依赖。选择你要运行的后端：

```bash
# 使用 .[vllm] 或 .[sglang] 来安装相应的框架依赖
uv pip install -e .
```

<Note>
仅执行基础的 `uv pip install -e .` 只会安装 Dynamo 运行时和前端。后端附加依赖（`[vllm]` 或 `[sglang]`）会安装运行推理 worker 所需的相应框架依赖。对于 TensorRT-LLM 后端，请改用 `tensorrtllm-runtime` 容器，而不是通过 `uv pip` 安装，以确保安装正确的依赖。更多详情请参阅[本地安装](local-installation.zh-CN.md)。
</Note>

## 8. 验证构建

```bash
python3 -m dynamo.frontend --help
```

你应该会看到 frontend 命令的帮助输出。

## DevContainer

VSCode 和 Cursor 用户可以使用预配置的开发容器，跳过手动设置。DevContainer 会自动安装所有工具链、构建项目，并设置 Python 环境。

针对 vLLM、SGLang 和 TensorRT-LLM 提供了特定框架的容器。有关设置说明，请参阅 [DevContainer README](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer)。

## 设置 Pre-commit Hooks

提交 PR 之前，请安装 pre-commit hooks，以确保你的代码通过 CI 检查：

```bash
uv pip install pre-commit
pre-commit install
```

手动对所有文件运行检查：

```bash
pre-commit run --all-files
```

## 故障排查

**缺少系统包**

如果 `maturin develop` 因链接器错误失败，请确认已安装所有系统依赖。在 Ubuntu 上：

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**虚拟环境未激活**

Maturin 会针对当前激活的 Python 解释器进行构建。如果你看到与 Python 或 site-packages 相关的错误，请确保虚拟环境已激活：

```bash
source .venv/bin/activate
```

**磁盘空间**

Rust 的 `target/` 目录在开发过程中可能增长到 10 GB 以上。如果构建因磁盘空间错误失败，请清理构建缓存：

```bash
cargo clean
```

**vLLM worker 启动失败：FlashInfer sampler 的 JIT 与 CUDA 13 wheels**

在 CUDA 13 源码安装环境下运行 vLLM worker 时，worker 可能在启动阶段因 FlashInfer JIT 错误而中止：

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

- [贡献指南](../contribution-guide.zh-CN.md) -- 贡献代码的工作流
- [示例](https://github.com/ai-dynamo/dynamo/tree/main/examples) -- 探索代码库
- [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) -- 查找可以着手处理的任务
