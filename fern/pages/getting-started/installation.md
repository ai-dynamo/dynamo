---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Installation

## Pip (PyPI)

Install a pre-built wheel from PyPI.

```bash
# Create a virtual environment and activate it
uv venv venv
source venv/bin/activate

# Install Dynamo from PyPI (choose one backend extra)
uv pip install "ai-dynamo[sglang]"  # or [vllm], [trtllm]
```

## Pip from source

Install directly from a local checkout for development.

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

# Create a virtual environment and activate it
uv venv venv
source venv/bin/activate
uv pip install ".[sglang]"  # or [vllm], [trtllm]
```

## Docker

Pull and run prebuilt images from NVIDIA NGC (`nvcr.io`).

```bash
# Run a container (mount your workspace if needed)
docker run --rm -it \
  --gpus all \
  --network host \
  nvcr.io/nvidia/ai-dynamo/sglang-runtime:latest  # or vllm, tensorrtllm
```
