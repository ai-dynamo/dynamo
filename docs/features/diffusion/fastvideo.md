---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FastVideo
subtitle: Deploys FastVideo text-to-video generation on Dynamo through a custom worker that serves the /v1/videos endpoint.
sidebar-title: FastVideo
---

This guide covers deploying [FastVideo](https://github.com/hao-ai-lab/FastVideo) text-to-video generation on Dynamo using a custom worker (`worker.py`) exposed through the `/v1/videos` endpoint.

> [!NOTE]
> Dynamo also supports diffusion through built-in backends: [SGLang Diffusion](../../backends/sglang/sglang-diffusion.md) (LLM diffusion, image, video), [vLLM-Omni](../../backends/vllm/vllm-omni.md) (text-to-image, text-to-video), and [TRT-LLM Diffusion](../../backends/trtllm/trtllm-diffusion.md) (text-to-image, text-to-video). See the [Diffusion Overview](README.md) for the full support matrix.

## Overview

- **Default model:** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`.
- **Typed API:** The worker builds `GeneratorConfig` once at startup and creates a `GenerationRequest` for each `/v1/videos` request.
- **Optimized inference:** `torch.compile` and FP4 quantization are available through `--torch-compile` and `--fp4-quantization`; the legacy `--enable-optimizations` flag remains as a shortcut for both.
- **Response format:** Returns one complete MP4 payload per request as `data[0].b64_json` (non-streaming).
- **Concurrency:** One request at a time per worker (VideoGenerator is not re-entrant). Scale throughput by running multiple workers.

> [!IMPORTANT]
> `worker.py` defaults to `--attention-backend VIDEO_SPARSE_ATTN` and routes `VSA_sparsity=0.8` through FastVideo's typed `PipelineSelection.experimental` config. Use `--attention-backend TORCH_SDPA` if your GPU/software stack does not support video sparse attention.

## Docker Image Build

The local Docker workflow builds a runtime image from the [`Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile):

- Base image: `nvidia/cuda:13.1.1-devel-ubuntu24.04`
- Installs [FastVideo](https://github.com/hao-ai-lab/FastVideo) from a pinned Git commit with `fastvideo.api`
- Installs the `ai-dynamo` package with `/v1/videos` support
- Compiles a [flash-attention](https://github.com/RandNMR73/flash-attention) fork from source

The Dockerfile exposes `TORCH_CUDA_ARCH_LIST` as a build argument (default: `10.0 10.0a` for Blackwell). Pass `--build-arg` to target a different architecture:

```bash
# Blackwell (default)
docker build examples/diffusers/ --build-arg TORCH_CUDA_ARCH_LIST="10.0 10.0a"

# Hopper
docker build examples/diffusers/ --build-arg TORCH_CUDA_ARCH_LIST="9.0 9.0a"
```

`MAX_JOBS` (default: `4`) controls parallel compilation jobs for flash-attention. Lower it if the build runs out of memory:

```bash
docker build examples/diffusers/ --build-arg MAX_JOBS=2
```

When using Docker Compose, set these as environment variables before running `docker compose up --build`:

```bash
# Hopper on a memory-constrained builder
TORCH_CUDA_ARCH_LIST="9.0 9.0a" MAX_JOBS=2 COMPOSE_PROFILES=4 docker compose up --build
```

> [!WARNING]
> The first Docker image build can take **20–40+ minutes** because FastVideo and CUDA-dependent components are compiled during the build. Subsequent builds are much faster if Docker layer cache is preserved. Compiling `flash-attention` can use significant RAM — low-memory builders may hit out-of-memory failures. If that happens, lower `MAX_JOBS` in the Dockerfile to reduce parallel compile memory usage. The [flash-attn install notes](https://pypi.org/project/flash-attn/) specifically recommend this on machines with less than 96 GB RAM and many CPU cores.

## Warmup Time

On first start, workers download model weights. When `--torch-compile` is enabled, compile/warmup steps can push the first ready time to roughly **10–20 minutes** (hardware-dependent). After the first successful compiled response, the second request can still take around **35 seconds** while runtime caches finish warming up; steady-state performance is typically reached from the third request onward.

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
| `MODEL` | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | HuggingFace model path |
| `NUM_GPUS` | `1` | Number of GPUs |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |
| `WORKER_EXTRA_ARGS` | — | Extra flags for `worker.py` (for example, `--torch-compile --fp4-quantization`) |
| `FRONTEND_EXTRA_ARGS` | — | Extra flags for `dynamo.frontend` |

Example:

```bash
MODEL=FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
NUM_GPUS=1 \
DYN_HTTP_PORT=8000 \
WORKER_EXTRA_ARGS="--torch-compile --fp4-quantization" \
./run_local.sh
```

> [!NOTE]
> FastVideo worker flags are not `dynamo.frontend` flags, so pass non-default worker configuration through `WORKER_EXTRA_ARGS`.

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
    "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "prompt": "A cinematic drone shot over a snowy mountain range at sunrise",
    "size": "256x256",
    "response_format": "b64_json",
    "nvext": {
      "fps": 8,
      "num_frames": 8,
      "num_inference_steps": 1,
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
| `--model`, `--model-path` | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | HuggingFace model path |
| `--num-gpus` | `1` | Number of GPUs for distributed inference |
| `--attention-backend` | `VIDEO_SPARSE_ATTN` | Sets `FASTVIDEO_ATTENTION_BACKEND`; choices: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `SAGE_SLA_ATTN` |
| `--vsa-sparsity` | `0.8` | Sets `PipelineSelection.experimental["VSA_sparsity"]` |
| `--torch-compile` | off | Enables FastVideo `CompileConfig` |
| `--fp4-quantization` | off | Requests FP4 transformer quantization through `QuantizationConfig` |
| `--enable-optimizations` | off | Backward-compatible shortcut for `--torch-compile --fp4-quantization` |
| `--dit-cpu-offload`, `--vae-cpu-offload`, `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--pin-cpu-memory` | on | CPU offload controls; each has a `--no-*` variant |
| `--max-video-width`, `--max-video-height` | `4096`, `4096` | Reject request dimensions above these caps before calling FastVideo |
| `--output-dir` | `$XDG_RUNTIME_DIR/dynamo-fastvideo/outputs` or `~/.cache/dynamo/fastvideo/outputs` | Directory for generated MP4 staging files |

### Request Parameters (`nvext`)

| Field | Default | Description |
|---|---|---|
| `fps` | `24` | Frames per second |
| `num_frames` | `125` | Total frames; overrides `fps * seconds` when set |
| `num_inference_steps` | `50` | Diffusion inference steps |
| `guidance_scale` | `1.0` | Classifier-free guidance scale |
| `seed` | `1024` | RNG seed for reproducibility |
| `negative_prompt` | — | Text to avoid in generation |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FASTVIDEO_VIDEO_CODEC` | `libx264` | Video codec for MP4 encoding |
| `FASTVIDEO_X264_PRESET` | `ultrafast` | x264 encoding speed preset |
| `FASTVIDEO_ATTENTION_BACKEND` | `VIDEO_SPARSE_ATTN` | Attention backend; `worker.py` sets this from `--attention-backend` and validates `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, and `SAGE_SLA_ATTN` |
| `FASTVIDEO_STAGE_LOGGING` | `1` | Enable per-stage timing logs |
| `FASTVIDEO_LOG_LEVEL` | — | Set to `DEBUG` for verbose logging |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM during Docker build | `flash-attention` compilation uses too much RAM | Pass `--build-arg MAX_JOBS=2` (or lower) at build time |
| `no kernel image available for this GPU` or CUDA arch error at runtime | Image was built for a different GPU architecture | Rebuild with the correct `TORCH_CUDA_ARCH_LIST` (e.g. `9.0 9.0a` for Hopper) |
| 10–20 min wait on first start with `--torch-compile` enabled | Model download + `torch.compile` warmup | Expected behavior; subsequent starts are faster if weights are cached |
| ~35 s second request | Runtime caches still warming | Steady-state performance from third request onward |
| Lower throughput than expected on B200/B300 | FP4/compile are opt-in | Pass `--torch-compile --fp4-quantization` |
| Startup or import failure after enabling FP4/compile or changing the attention backend | FP4 and some attention backends depend on specific hardware/software support | Re-run `worker.py` without `--torch-compile --fp4-quantization`, or use `--attention-backend TORCH_SDPA` |

## Source Code

The example source lives at [`examples/diffusers/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers) in the Dynamo repository.

## See Also

- [vLLM-Omni Text-to-Video](../../backends/vllm/vllm-omni.md#text-to-video) — vLLM-Omni video generation via `/v1/videos`
- [vLLM-Omni Text-to-Image](../../backends/vllm/vllm-omni.md#text-to-image) — vLLM-Omni image generation
- [SGLang Video Generation](../../backends/sglang/sglang-diffusion.md#video-generation) — SGLang video generation worker
- [SGLang Image Diffusion](../../backends/sglang/sglang-diffusion.md#image-diffusion) — SGLang image diffusion worker
- [TRT-LLM Diffusion](../../backends/trtllm/trtllm-diffusion.md#quick-start) — TensorRT-LLM diffusion quick start
- [Diffusion Overview](README.md) — Full backend support matrix
