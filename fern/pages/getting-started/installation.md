---
title: "Installation"
---

{/*
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/}

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
