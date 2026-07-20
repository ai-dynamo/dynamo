---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Local Installation
sidebar-title: Local Installation
description: Install and run Dynamo on a local machine or VM with containers or PyPI
---

This guide walks through installing and running Dynamo on a local machine or VM with one or more GPUs. By the end, you'll have a working OpenAI-compatible endpoint serving a model.

For production multi-node clusters, see the [Kubernetes Deployment Guide](../kubernetes/README.md). To build from source for development, see [Building from Source](building-from-source.md).

## System Requirements

<Tabs>
  <Tab title="CUDA">

    | Requirement | Supported |
    |---|---|
    | **GPU** | NVIDIA Ampere, Ada Lovelace, Hopper, Blackwell |
    | **OS** | Ubuntu 22.04, Ubuntu 24.04 |
    | **Architecture** | x86_64, ARM64 (ARM64 requires Ubuntu 24.04) |
    | **CUDA** | 12.9+ or 13.0+ (B300/GB300 require CUDA 13) |
    | **Python** | 3.10, 3.12 |
    | **Driver** | 575.51.03+ (CUDA 12) or 580.00.03+ (CUDA 13) |

    TensorRT-LLM does not support Python 3.11.

  </Tab>
  <Tab title="XPU">

    | Requirement | Supported |
    |---|---|
    | **XPU** | Intel® Data Center GPU Max Series, Intel® Arc™ Pro B-Series Graphics Cards |
    | **OS** | Ubuntu 24.04 |
    | **Architecture** | x86_64 |
    | **Python** | 3.12 |
    | **oneAPI** | 2025.3 |
    | **Driver** | 25.48.36300.8 |

  </Tab>
</Tabs>

For the full compatibility matrix including backend framework versions, see the [Support Matrix](../reference/support-matrix.md).

## Install Dynamo

### Option A: Containers (Recommended)

<Tabs>
  <Tab title="CUDA">

    Containers have all dependencies pre-installed. No setup required.

    ```bash
    # SGLang
    docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1

    # TensorRT-LLM
    docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1

    # vLLM
    docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1
    ```

  </Tab>
  <Tab title="XPU">

    Build XPU runtime images from this repository. `container/render.py` supports `--device=xpu` for vLLM and SGLang runtime images. `container/run.sh --device=xpu` exposes `/dev/dri` and the host `render` group to the container.

    **vLLM**

    ```bash
    git clone https://github.com/ai-dynamo/dynamo.git
    cd dynamo
    container/render.py --framework=vllm --device=xpu --target=runtime
    docker build -t dynamo:latest-vllm-xpu-runtime \
      -f container/vllm-runtime-xpu-amd64-rendered.Dockerfile .
    container/run.sh --image dynamo:latest-vllm-xpu-runtime --device=xpu -it
    ```

    **SGLang**

    ```bash
    git clone https://github.com/ai-dynamo/dynamo.git
    cd dynamo
    container/render.py --framework=sglang --device=xpu --target=runtime
    docker build -t dynamo:latest-sglang-xpu-runtime \
      -f container/sglang-runtime-xpu-amd64-rendered.Dockerfile .
    container/run.sh --image dynamo:latest-sglang-xpu-runtime --device=xpu -it
    ```

  </Tab>
</Tabs>

To run frontend and worker in the same container, either:

- Run processes in background with `&` (see Run Dynamo section below), or
- Open a second terminal and use `docker exec -it <container_id> bash`

See [Release Artifacts](../reference/release-artifacts.md#container-images) for available
versions and backend guides for run instructions: [SGLang](../backends/sglang/README.md) |
[TensorRT-LLM](../backends/trtllm/README.md) | [vLLM](../backends/vllm/README.md)

### Option B: Install from PyPI

<Tabs>
  <Tab title="CUDA">

    Supported for vLLM and SGLang only. Use Option A for TensorRT-LLM.

    ```bash
    # Install uv (recommended Python package manager)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create virtual environment
    uv venv venv
    source venv/bin/activate
    uv pip install pip
    ```

    Install system dependencies and the Dynamo wheel for your chosen backend:

    **SGLang**

    ```bash
    sudo apt install python3-dev
    uv pip install --prerelease=allow "ai-dynamo[sglang]"
    ```

    **vLLM**

    ```bash
    sudo apt install python3-dev libxcb1
    uv pip install --prerelease=allow "ai-dynamo[vllm]"
    ```

  </Tab>
  <Tab title="XPU">

    The PyPI backend extras are CUDA-only. For XPU, use the source-built container path in [Option A: Containers](#option-a-containers-recommended).

  </Tab>
</Tabs>

## Run Dynamo

### Discovery Backend

Dynamo components discover each other through a shared backend. Two options are available:

| Backend | When to Use | Setup |
|---|---|---|
| **File** | Single machine, local development | No setup -- pass `--discovery-backend file` to all components. The event plane automatically defaults to ZMQ (no NATS required). |
| **etcd** | Multi-node, production | Requires a running etcd instance (default if no flag is specified). The event plane defaults to NATS. |

This guide uses `--discovery-backend file`. For etcd setup, see [Service Discovery](../kubernetes/service-discovery.md).

### Verify Installation (Optional)

Verify the CLI is installed and callable:

```bash
python3 -m dynamo.frontend --help
```

If you cloned the repository, you can run additional system checks:

```bash
python3 dev/sanity_check.py
```

### Start the Frontend

```bash
# Start the OpenAI compatible frontend (default port is 8000)
python3 -m dynamo.frontend --discovery-backend file
```

To run in a single terminal (useful in containers), append `> logfile.log 2>&1 &`
to run processes in background:

```bash
python3 -m dynamo.frontend --discovery-backend file > dynamo.frontend.log 2>&1 &
```

### Start a Worker

In another terminal (or same terminal if using background mode), start a worker for your chosen accelerator and backend:

<Tabs>
  <Tab title="CUDA">

    **SGLang**

    ```bash
    python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
    ```

    **TensorRT-LLM**

    ```bash
    python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file
    ```

    The warning `Cannot connect to ModelExpress server/transport error. Using direct download.`
    is expected in this local single-machine setup (no ModelExpress server running) and can
    be safely ignored. In a Kubernetes deployment where `MODEL_EXPRESS_URL` is configured,
    this warning -- or the related `Failed to resolve local model path after server download`
    -- indicates that ModelExpress is configured but is not actually serving cached models;
    see [Model Caching in Kubernetes](../kubernetes/model-caching.md#option-2-modelexpress-p2p-distribution)
    for the correct configuration.

    **vLLM**

    ```bash
    python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file
    ```

  </Tab>
  <Tab title="XPU">

    **vLLM**

    ```bash
    VLLM_TARGET_DEVICE=xpu \
      python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file
    ```

    **SGLang**

    ```bash
    python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
    ```

  </Tab>
</Tabs>

## Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```

## Troubleshooting

<Tabs>
  <Tab title="CUDA">

    **CUDA/driver version mismatch**

    Run `nvidia-smi` to check your driver version. Dynamo requires driver 575.51.03+ for CUDA 12 or 580.00.03+ for CUDA 13. B300/GB300 GPUs require CUDA 13. See the [Support Matrix](../reference/support-matrix.md) for full requirements.

    **Model doesn't fit on GPU (OOM)**

    The default model `Qwen/Qwen3-0.6B` requires ~2GB of GPU memory. Larger models need more VRAM:

    | Model Size | Approximate VRAM |
    |---|---|
    | 7B | 14-16 GB |
    | 13B | 26-28 GB |
    | 70B | 140+ GB (multi-GPU) |

    Start with a small model and scale up based on your hardware.

    **TensorRT-LLM**

    TensorRT-LLM is not supported via a local PyPI install. Use the
    `tensorrtllm-runtime` container (Option A).

    **Container runs but GPU not detected**

    Pass `--gpus all` to `docker run`. Without this flag, the container won't have access to NVIDIA GPUs:

    ```bash
    # Correct
    docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1

    # Wrong -- no GPU access
    docker run --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1
    ```

    **vLLM worker fails to start: FlashInfer sampler JIT and CUDA 13 wheels**

    When you run a vLLM worker from a CUDA 13 install, the worker can abort during startup with a FlashInfer JIT error:

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

  </Tab>
  <Tab title="XPU">

    **XPU/driver version mismatch**

    Use the XPU Driver 25.48.36300.8 and oneAPI 2025.3 versions listed in [System Requirements](#system-requirements). Rebuild the XPU runtime image after changing the driver stack.

    **Model doesn't fit on XPU memory (OOM)**

    The default model `Qwen/Qwen3-0.6B` requires ~2GB of device memory. Larger models need more memory:

    | Model Size | Approximate memory |
    |---|---|
    | 7B | 14-16 GB |
    | 13B | 26-28 GB |
    | 70B | 140+ GB |

    Start with a small model and scale up based on your XPU capacity.

    **Container runs but XPU is not detected**

    Use `container/run.sh --device=xpu`. The wrapper adds `/dev/dri` and the host `render` group for XPU access.

    ```bash
    container/run.sh --image dynamo:latest-vllm-xpu-runtime --device=xpu -it
    ```

    Check that the host exposes `/dev/dri` and has a `render` group:

    ```bash
    ls -l /dev/dri
    getent group render
    ```

  </Tab>
</Tabs>

## Next Steps

- [Backend Guides](../backends/sglang/README.md) -- Backend-specific configuration and features
- [Disaggregated Serving](../features/disaggregated-serving/README.md) -- Scale prefill and decode independently
- [KV Cache Aware Routing](../components/router/router-guide.md) -- Smart request routing
- [Kubernetes Deployment](../kubernetes/README.md) -- Production multi-node deployments
