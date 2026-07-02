---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Building from Source
sidebar-title: Building from Source
description: Build Dynamo from source for development and contributions
---

[简体中文](./building-from-source.zh-CN.md)

Build Dynamo from source when you want to contribute code, test features on the development branch, or customize the build. If you just want to run Dynamo, the [Local Installation](local-installation.md) guide is faster.

This guide covers Ubuntu and macOS. For a containerized dev environment that handles all of this automatically, see [DevContainer](#devcontainer).

## 1. Install System Libraries

**Ubuntu:**

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake protobuf

# Verify Metal is accessible
xcrun -sdk macosx metal
```

## 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## 3. Create a Python Virtual Environment

Install [uv](https://docs.astral.sh/uv/#installation) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

## 4. Install Build Tools

```bash
uv pip install pip 'maturin[patchelf]'
```

[Maturin](https://github.com/PyO3/maturin) is the Rust-Python bindings build tool. The `patchelf` extra lets maturin patch native extension library paths during the build.

## 5. Build the Rust Bindings

```bash
cd lib/bindings/python
maturin develop --uv
```

## 6. Install GPU Memory Service

```bash
# Return to project root
cd "$(git rev-parse --show-toplevel)"
uv pip install -e lib/gpu_memory_service
```

## 7. Install the Wheel

Install Dynamo with a backend extra to pull the inference engine and its CUDA dependencies. Choose the backend you intend to run:

```bash
uv pip install -e ".[vllm]"   # or [sglang]
```

> [!NOTE]
> The base `uv pip install -e .` installs only the Dynamo runtime and frontend. A backend extra (`[vllm]`, `[sglang]`, or `[trtllm]`) will install the relevant framework dependencies to run an inference worker. TensorRT-LLM cannot be installed with `uv` — it needs `pip` with the NVIDIA package index (`uv` cannot resolve a transitive Git URL dependency). See [Local Installation](local-installation.md) or the [TRT-LLM backend guide](../backends/trtllm/README.md).

## 8. Verify the Build

```bash
python3 -m dynamo.frontend --help
```

You should see the frontend command help output.

## DevContainer

VSCode and Cursor users can skip manual setup using pre-configured development containers. The DevContainer installs all toolchains, builds the project, and sets up the Python environment automatically.

Framework-specific containers are available for vLLM, SGLang, and TensorRT-LLM. See the [DevContainer README](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer) for setup instructions.

## Set Up Pre-commit Hooks

Before submitting PRs, install the pre-commit hooks to ensure your code passes CI checks:

```bash
uv pip install pre-commit
pre-commit install
```

Run checks manually on all files:

```bash
pre-commit run --all-files
```

## Troubleshooting

**Missing system packages**

If `maturin develop` fails with linker errors, verify all system dependencies are installed. On Ubuntu:

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**Virtual environment not activated**

Maturin builds against the active Python interpreter. If you see errors about Python or site-packages, ensure your virtual environment is activated:

```bash
source .venv/bin/activate
```

**Disk space**

The Rust `target/` directory can grow to 10+ GB during development. If builds fail with disk space errors, clean the build cache:

```bash
cargo clean
```

**vLLM worker fails to start: FlashInfer sampler JIT and CUDA 13 wheels**

When you run a vLLM worker from a CUDA 13 source install, the worker can abort during startup with a FlashInfer JIT error:

```text
RuntimeError: Engine core initialization failed.
...
cuda/std/__cccl/cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are incompatible"
```

The CUDA wheels resolved for a CUDA 13 install can be version-skewed: `torch` pins the runtime headers to 13.0, while vLLM's `tilelang` dependency pulls `nvidia-cuda-nvcc` 13.2. FlashInfer compiles its sampler kernel with `nvcc` against those headers, and the version mismatch fails the build. This is tracked upstream at [flashinfer#3493](https://github.com/flashinfer-ai/flashinfer/issues/3493).

Set `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM falls back to its native sampler:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Next Steps

- [Contribution Guide](../contribution-guide.md) -- Workflow for contributing code
- [Examples](https://github.com/ai-dynamo/dynamo/tree/main/examples) -- Explore the codebase
- [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) -- Find a task to work on
