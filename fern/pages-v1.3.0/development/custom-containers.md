---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Runtime Containers
subtitle: Build Dynamo runtime images for built-in or custom backends
---

Dynamo runtime images package the Dynamo runtime with an inference engine. The same container build flow can generate images for the built-in engines or a backend that you add on top of the Dynamo runtime.

Use [`container/render.py`](../../container/render.py) to select the engine family and Docker target:

```bash
# vLLM runtime image
python container/render.py --framework=vllm --target=runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .

# SGLang runtime image
python container/render.py --framework=sglang --target=runtime --output-short-filename
docker build -t dynamo:latest-sglang-runtime -f container/rendered.Dockerfile .

# TensorRT-LLM runtime image
python container/render.py --framework=trtllm --target=runtime --cuda-version=13.1 --output-short-filename
docker build -t dynamo:latest-trtllm-runtime -f container/rendered.Dockerfile .
```

## Engine and Target Toggles

`--framework` chooses the engine base. Use `vllm`, `sglang`, or `trtllm` for built-in backends. Use `none` when you want a Dynamo-only base image and plan to install your own backend package.

`--target` chooses the image shape:

| Target | Use when |
| --- | --- |
| `runtime` | Running inference, benchmarks, or Kubernetes deployments. |
| `local-dev` | Developing locally with the workspace bind-mounted into the container. |
| `dev` | Legacy root-based development workflows. Prefer `local-dev` for new work. |

## Custom Backend Image

For a Python custom backend, start with a built-in engine image if you need that framework's CUDA/Python stack, or use `--framework=none` if your backend brings its own dependencies:

```bash
python container/render.py --framework=none --target=runtime --output-short-filename
docker build -t dynamo:custom-backend-base -f container/rendered.Dockerfile .
```

Then layer your backend package into a small Dockerfile:

```Dockerfile
FROM dynamo:custom-backend-base

COPY dist/my_backend-*.whl /tmp/
RUN uv pip install --system --no-deps /tmp/my_backend-*.whl

ENTRYPOINT ["my-backend"]
```

For a Rust custom backend, build the backend binary in your own builder stage and copy it into the Dynamo runtime image:

```Dockerfile
FROM rust:1.96.1 AS backend-builder
WORKDIR /src
COPY . .
RUN cargo build --release

FROM dynamo:custom-backend-base
COPY --from=backend-builder /src/target/release/my-backend /usr/local/bin/my-backend

ENTRYPOINT ["my-backend"]
```

## Run Locally

Use `container/run.sh` to launch the image with the same GPU and mount defaults used by Dynamo development workflows:

```bash
container/run.sh --image dynamo:custom-backend-base --mount-workspace -it
```

For the full container build reference, target matrix, and troubleshooting notes, see the repository-level [Container Development Guide](../../container/README.md).
