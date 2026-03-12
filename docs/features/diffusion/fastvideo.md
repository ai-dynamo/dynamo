---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: FastVideo
---

# FastVideo

This guide covers deploying [FastVideo](https://github.com/hao-ai-lab/FastVideo) text-to-video generation on Dynamo using a custom worker (`worker.py`) exposed through the `/v1/videos` endpoint.

> [!NOTE]
> Dynamo also supports diffusion through built-in backends: [SGLang Diffusion](../../backends/sglang/sglang-diffusion.md) (LLM diffusion, image, video), [vLLM-Omni](../../backends/vllm/vllm-omni.md) (text-to-image, text-to-video), and [TRT-LLM Video Diffusion](../../backends/trtllm/trtllm-video-diffusion.md). See the [Diffusion Overview](README.md) for the full support matrix.

## Overview

- **Default model:** `FastVideo/LTX2-Distilled-Diffusers` — a distilled variant of the LTX-2 Diffusion Transformer (Lightricks), reducing inference from 50+ steps to just 5.
- **Two-stage pipeline:** Stage 1 generates video at target resolution; Stage 2 refines with a distilled LoRA for improved fidelity and texture.
- **Optimized inference:** FP4 quantization and `torch.compile` are enabled by default for maximum throughput.
- **Response format:** Returns one complete MP4 payload per request as `data[0].b64_json` (non-streaming).
- **Concurrency:** One request at a time per worker (VideoGenerator is not re-entrant). Scale throughput by running multiple workers.

> [!IMPORTANT]
> This example is optimized for **NVIDIA B200/B300** GPUs (CUDA arch 10.0) with FP4 quantization and flash-attention. It can run on other GPUs (H100, A100, etc.) by passing `--disable-optimizations` to `worker.py`, which disables FP4 quantization, `torch.compile`, and switches the attention backend from FLASH_ATTN to TORCH_SDPA. Expect lower performance but broader compatibility.

## Docker Image Build

The local Docker workflow builds a runtime image from the [`Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile):

- Base image: `nvidia/cuda:13.1.1-devel-ubuntu24.04`
- Installs [FastVideo](https://github.com/hao-ai-lab/FastVideo) from GitHub
- Installs Dynamo from the `release/1.0.0` branch (for `/v1/videos` support)
- Compiles a [flash-attention](https://github.com/RandNMR73/flash-attention) fork from source

> [!WARNING]
> The first Docker image build can take **20–40+ minutes** because FastVideo and CUDA-dependent components are compiled during the build. Subsequent builds are much faster if Docker layer cache is preserved. Compiling `flash-attention` can use significant RAM — low-memory builders may hit out-of-memory failures. If that happens, lower `MAX_JOBS` in the Dockerfile to reduce parallel compile memory usage. The [flash-attn install notes](https://pypi.org/project/flash-attn/) specifically recommend this on machines with less than 96 GB RAM and many CPU cores.

## Warmup Time

On first start, workers download model weights and run compile/warmup steps. Expect roughly **10–20 minutes** before the first request is ready (hardware-dependent). After the first successful response, the second request can still take around **35 seconds** while runtime caches finish warming up; steady-state performance is typically reached from the third request onward.

> [!TIP]
> When using Kubernetes, mount a shared Hugging Face cache PVC (see [Kubernetes Deployment](#kubernetes-deployment)) so model weights are downloaded once and reused across pod restarts.

## Local Deployment

### Prerequisites

**For Docker Compose:**

- Docker Engine 26.0+
- Docker Compose v2
- NVIDIA Container Toolkit

**For host-local script:**

- Python environment with Dynamo + FastVideo dependencies installed
- CUDA-compatible GPU runtime available on host

### Option 1: Docker Compose

```bash
cd <dynamo-root>/examples/diffusers/local

# Start 4 workers on GPUs 0..3
COMPOSE_PROFILES=4 docker compose up --build
```

The Compose file builds from the Dockerfile and exposes the API on `http://localhost:8000`. See the [Docker Image Build](#docker-image-build) section for build time expectations.

### Option 2: Host-Local Script

```bash
cd <dynamo-root>/examples/diffusers/local
./run_local.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `python3` | Python interpreter |
| `MODEL` | `FastVideo/LTX2-Distilled-Diffusers` | HuggingFace model path |
| `NUM_GPUS` | `1` | Number of GPUs |
| `HTTP_PORT` | `8000` | Frontend HTTP port |
| `WORKER_EXTRA_ARGS` | — | Extra flags for `worker.py` (e.g., `--disable-optimizations`) |
| `FRONTEND_EXTRA_ARGS` | — | Extra flags for `dynamo.frontend` |

Example:

```bash
MODEL=FastVideo/LTX2-Distilled-Diffusers \
NUM_GPUS=1 \
HTTP_PORT=8000 \
WORKER_EXTRA_ARGS="--disable-optimizations" \
./run_local.sh
```

> [!NOTE]
> `--disable-optimizations` is a `worker.py` flag (not a `dynamo.frontend` flag), so pass it through `WORKER_EXTRA_ARGS`.

The script writes logs to:

- `.runtime/logs/worker.log`
- `.runtime/logs/frontend.log`

## Kubernetes Deployment

### Files

| File | Description |
|---|---|
| `agg.yaml` | Base aggregated deployment (Frontend + `FastVideoWorker`) |
| `agg_user_workload.yaml` | Same deployment with `user-workload` tolerations and `imagePullSecrets` |
| `huggingface-cache-pvc.yaml` | Shared HF cache PVC for model weights |
| `dynamo-platform-values-user-workload.yaml` | Optional Helm values for clusters with tainted `user-workload` nodes |

### Prerequisites

1. Dynamo Kubernetes Platform installed
2. GPU-enabled Kubernetes cluster
3. FastVideo runtime image pushed to your registry
4. Optional HF token secret (for gated models)

Create a Hugging Face token secret if needed:

```bash
export NAMESPACE=<your-namespace>
export HF_TOKEN=<your-hf-token>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### Deploy

```bash
cd <dynamo-root>/examples/diffusers/deploy
export NAMESPACE=<your-namespace>

kubectl apply -f huggingface-cache-pvc.yaml -n ${NAMESPACE}
kubectl apply -f agg.yaml -n ${NAMESPACE}
```

For clusters with tainted `user-workload` nodes and private registry pulls:

1. Set your pull secret name and image in `agg_user_workload.yaml`.
2. Apply:

```bash
kubectl apply -f huggingface-cache-pvc.yaml -n ${NAMESPACE}
kubectl apply -f agg_user_workload.yaml -n ${NAMESPACE}
```

### Update Image Quickly

```bash
export DEPLOYMENT_FILE=agg.yaml
export FASTVIDEO_IMAGE=<my-registry/fastvideo-runtime:my-tag>

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(FASTVIDEO_IMAGE)' \
  ${DEPLOYMENT_FILE} > ${DEPLOYMENT_FILE}.generated

kubectl apply -f ${DEPLOYMENT_FILE}.generated -n ${NAMESPACE}
```

### Verify and Access

```bash
kubectl get dgd -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l nvidia.com/dynamo-component=FastVideoWorker
```

```bash
kubectl port-forward -n ${NAMESPACE} svc/fastvideo-agg-frontend 8000:8000
```

## Test Request

> [!NOTE]
> If this is the first request after startup, expect it to take longer while warmup completes. See [Warmup Time](#warmup-time) for details.

Send a request and decode the response:

```bash
curl -s -X POST http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "FastVideo/LTX2-Distilled-Diffusers",
    "prompt": "A cinematic drone shot over a snowy mountain range at sunrise",
    "size": "1920x1088",
    "seconds": 5,
    "nvext": {
      "fps": 24,
      "num_frames": 121,
      "num_inference_steps": 5,
      "guidance_scale": 1.0,
      "seed": 10
    }
  }' > response.json

# Linux
jq -r '.data[0].b64_json' response.json | base64 --decode > output.mp4

# macOS
jq -r '.data[0].b64_json' response.json | base64 -D > output.mp4
```

## Worker Configuration Reference

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `FastVideo/LTX2-Distilled-Diffusers` | HuggingFace model path |
| `--num-gpus` | `1` | Number of GPUs for distributed inference |
| `--disable-optimizations` | off | Disables FP4 quantization, `torch.compile`, and switches attention from FLASH_ATTN to TORCH_SDPA |

### Request Parameters (`nvext`)

| Field | Default | Description |
|---|---|---|
| `fps` | `24` | Frames per second |
| `num_frames` | `121` | Total frames; overrides `fps * seconds` when set |
| `num_inference_steps` | `5` | Diffusion inference steps |
| `guidance_scale` | `1.0` | Classifier-free guidance scale |
| `seed` | `10` | RNG seed for reproducibility |
| `negative_prompt` | — | Text to avoid in generation |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FASTVIDEO_VIDEO_CODEC` | `libx264` | Video codec for MP4 encoding |
| `FASTVIDEO_X264_PRESET` | `ultrafast` | x264 encoding speed preset |
| `FASTVIDEO_ATTENTION_BACKEND` | `FLASH_ATTN` | Attention backend (`FLASH_ATTN` or `TORCH_SDPA`) |
| `FASTVIDEO_STAGE_LOGGING` | `1` | Enable per-stage timing logs |
| `FASTVIDEO_LOG_LEVEL` | — | Set to `DEBUG` for verbose logging |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM during Docker build | `flash-attention` compilation uses too much RAM | Lower `MAX_JOBS` in the Dockerfile |
| 10–20 min wait on first start | Model download + `torch.compile` warmup | Expected behavior; subsequent starts are faster if weights are cached |
| ~35 s second request | Runtime caches still warming | Steady-state performance from third request onward |
| Poor performance on non-B200/B300 GPUs | FP4 and flash-attention optimizations require CUDA arch 10.0 | Pass `--disable-optimizations` to `worker.py` |

## Source Code

The example source lives at [`examples/diffusers/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers) in the Dynamo repository.

## See Also

- [vLLM-Omni Text-to-Video](../../backends/vllm/vllm-omni.md#text-to-video) — vLLM-Omni video generation via `/v1/videos`
- [vLLM-Omni Text-to-Image](../../backends/vllm/vllm-omni.md#text-to-image) — vLLM-Omni image generation
- [SGLang Video Generation](../../backends/sglang/sglang-diffusion.md#video-generation) — SGLang video generation worker
- [SGLang Image Diffusion](../../backends/sglang/sglang-diffusion.md#image-diffusion) — SGLang image diffusion worker
- [TRT-LLM Video Diffusion](../../backends/trtllm/trtllm-video-diffusion.md#quick-start) — TensorRT-LLM video diffusion quick start
- [Diffusion Overview](README.md) — Full backend support matrix
