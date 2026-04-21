---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Building from Source
description: Build Dynamo from source for development and contributions
---

# Building from Source

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
uv pip install pip maturin
```

[Maturin](https://github.com/PyO3/maturin) is the Rust-Python bindings build tool.

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

```bash
uv pip install -e .
```

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

## Next Steps

- [Contribution Guide](../contribution-guide.md) -- Workflow for contributing code
- [Examples](https://github.com/ai-dynamo/dynamo/tree/main/examples) -- Explore the codebase
- [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) -- Find a task to work on
