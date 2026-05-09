---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Building a Custom TensorRT-LLM Container
---

For the prebuilt container, see the [TensorRT-LLM Quick Start](README.md#quick-start).

## Building a Custom Container

If you need to build a container from source (e.g., for custom modifications or a different CUDA version):

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
python container/render.py --framework=trtllm --target=runtime --output-short-filename --cuda-version=13.1
docker build -t dynamo:trtllm-latest -f container/rendered.Dockerfile .

# On an ARM machine:
python container/render.py --framework=trtllm --target=runtime --platform=arm64 --output-short-filename --cuda-version=13.1
docker build -t dynamo:trtllm-latest -f container/rendered.Dockerfile .
```

Run the custom container:

```bash
./container/run.sh --framework trtllm -it
```
