---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: FastVideo
---

# FastVideo

This guide covers deploying [FastVideo](https://github.com/hao-ai-lab/FastVideo) text-to-video generation on Dynamo as a built-in backend via `python -m dynamo.fastvideo`, exposed through the `/v1/videos` endpoint.

> [!NOTE]
> Dynamo also supports diffusion through built-in backends: [SGLang Diffusion](../../backends/sglang/sglang-diffusion.md) (LLM diffusion, image, video), [vLLM-Omni](../../backends/vllm/vllm-omni.md) (text-to-image, text-to-video), and [TRT-LLM Video Diffusion](../../backends/trtllm/trtllm-video-diffusion.md). See the [Diffusion Overview](README.md) for the full support matrix.

## Overview

- **Default model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` — a single-GPU-friendly Wan 2.1 text-to-video model that keeps the built-in backend examples and smoke tests approachable.
- **Alternate models:** You serve other FastVideo-compatible checkpoints, including `FastVideo/LTX2-Distilled-Diffusers`, by overriding `--model`.
- **Optimized inference:** The built-in backend exposes explicit runtime flags such as `--torch-compile`, `--fp4-quantization`, `--attention-backend`, and CPU offload controls instead of bundling them into a single profile.
- **Response format:** Uses Dynamo's shared video protocol types and returns one complete MP4 payload per request. By default, responses use `data[0].url` and store output via Dynamo media storage. Set `"response_format": "b64_json"` when you want inline base64 video data instead.
- **Concurrency:** One request at a time per worker (VideoGenerator is not re-entrant). Scale throughput by running multiple workers.
- **Deployment mode:** FastVideo currently supports aggregated deployment only. Disaggregated serving is not supported yet.

> [!IMPORTANT]
> `dynamo.fastvideo` defaults to `--attention-backend TORCH_SDPA` for broader compatibility across GPUs, including systems such as H100. On Blackwell GPUs, the optimized path is typically `--torch-compile --fp4-quantization`; if desired, you can also opt into flash-attention explicitly with `--attention-backend FLASH_ATTN`.

## Direct Launch

Launch the built-in backend directly:

```bash
python -m dynamo.fastvideo --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

The local example flow in `examples/backends/fastvideo/launch/` now shells into this built-in entrypoint rather than maintaining separate worker logic.

## Docker Image Build

The canonical FastVideo runtime image is built through the shared container renderer so CI, release, and local image generation all follow the same backend path:

```bash
python3 container/render.py \
  --framework fastvideo \
  --target runtime \
  --cuda-version 13.1 \
  --platform linux/amd64

docker build -f container/fastvideo-runtime-cuda13.1-amd64-rendered.Dockerfile .
```

For source-based container development, the same renderer also supports
`--target dev` and `--target local-dev` for `--framework fastvideo`. For
lower-level container work, FastVideo also exposes the same intermediate
renderer targets as the other runtime backends: `--target base` and
`--target wheel_builder`.

This rendered runtime image:

- Uses a multi-stage build with `nvcr.io/nvidia/pytorch:25.12-py3` as the build-side base
- Publishes the final runtime stage from `nvcr.io/nvidia/cuda-dl-base:25.12-cuda13.1-runtime-ubuntu24.04`
- Installs the local Dynamo runtime wheels produced by the shared wheel builder
- Installs [FastVideo](https://github.com/hao-ai-lab/FastVideo) and related Python dependencies in a dedicated framework stage through the shared `container/deps/fastvideo/` installer path, then copies the prepared environment into the final runtime image

Released FastVideo runtime images are also published to NGC for local container use:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/fastvideo-runtime:1.0.1-cuda13
```

## Warmup Time

On first start, workers download model weights. When `--torch-compile` is enabled, compile and warmup steps can push the first ready time to roughly **10–20 minutes** (hardware-dependent). After the first successful optimized response, the second request can still take around **35 seconds** while runtime caches finish warming up; steady-state performance is typically reached from the third request onward.

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
cd <dynamo-root>/examples/backends/fastvideo/launch

docker pull nvcr.io/nvidia/ai-dynamo/fastvideo-runtime:1.0.1-cuda13

# Start 4 workers on GPUs 0..3
COMPOSE_PROFILES=4 docker compose up
```

The Compose file defaults to `nvcr.io/nvidia/ai-dynamo/fastvideo-runtime:1.0.1-cuda13` and exposes the API on `http://localhost:8000`. Override the image for top-of-tree or private builds with `FASTVIDEO_IMAGE=<custom-image> COMPOSE_PROFILES=4 docker compose up`. The Compose example starts a local `etcd` sidecar for service discovery, which avoids host filesystem ownership issues between the frontend and worker containers.

By default, the Compose example downloads model weights into the container's
own writable cache directory. If you want to reuse an existing local Hugging
Face cache across container recreations, add a bind mount in
`examples/backends/fastvideo/launch/docker-compose.yml`, for example:

```yaml
- ${HOME}/.cache/huggingface:/home/dynamo/.cache/huggingface
```

The host cache directory must be writable by UID `1000` inside the container
(`dynamo`).

### Option 2: Host-Local Script

```bash
cd <dynamo-root>/examples/backends/fastvideo/launch
./run_local.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `python3` | Python interpreter |
| `MODEL` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | HuggingFace model path |
| `NUM_GPUS` | `1` | Number of GPUs |
| `HTTP_PORT` | `8000` | Frontend HTTP port |
| `WORKER_EXTRA_ARGS` | — | Extra flags for `dynamo.fastvideo` (for example, `--torch-compile --fp4-quantization --attention-backend FLASH_ATTN`) |
| `FRONTEND_EXTRA_ARGS` | — | Extra flags for `dynamo.frontend` |

Example:

```bash
MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
NUM_GPUS=1 \
HTTP_PORT=8000 \
WORKER_EXTRA_ARGS="--torch-compile --fp4-quantization --attention-backend FLASH_ATTN" \
./run_local.sh
```

> [!NOTE]
> `--torch-compile`, `--fp4-quantization`, and `--attention-backend` are `dynamo.fastvideo` flags, not `dynamo.frontend` flags, so pass them through `WORKER_EXTRA_ARGS` when you want a non-default backend configuration.

Use the built-in entrypoint directly if you prefer not to use the wrapper script:

```bash
PYTHONPATH=<dynamo-root>/components/src \
python -m dynamo.fastvideo \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --discovery-backend file
```

The script writes logs to:

- `.runtime/logs/worker.log`
- `.runtime/logs/frontend.log`

## Kubernetes Deployment

### Files

| File | Description |
|---|---|
| `agg.yaml` | Base aggregated deployment (Frontend + `FastVideoWorker`) |

> [!NOTE]
> FastVideo currently has only an aggregated deployment path in Dynamo. The example manifests in this directory do not include a disaggregated FastVideo setup.

### Prerequisites

1. Dynamo Kubernetes Platform installed
2. GPU-enabled Kubernetes cluster
3. FastVideo runtime image pushed to your registry
4. `hf-token-secret` Kubernetes secret for model access

Create the Hugging Face token secret:

```bash
export NAMESPACE=<your-namespace>
export HF_TOKEN=<your-hf-token>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### Deploy

```bash
cd <dynamo-root>/examples/backends/fastvideo/deploy
export NAMESPACE=<your-namespace>

kubectl apply -f agg.yaml -n ${NAMESPACE}
```

The example deployment manifests use the backend default
(`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) for faster startup and single-GPU
compatibility. To use another FastVideo-compatible model such as
`FastVideo/LTX2-Distilled-Diffusers`, update the worker `--model`
argument in the manifest.

The shipped Kubernetes manifests also set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and add the
`SYS_PTRACE` container capability on the FastVideo worker. These match the
FastVideo CI runtime requirements: the allocator setting reduces single-GPU OOM
pressure during model load, and `SYS_PTRACE` avoids `pidfd_getfd` failures from
FastVideo's CUDA IPC path. Unlike the local Docker Compose setup, the
Kubernetes example manifests do not override TorchInductor or Triton cache
paths; they rely on the backend's default writable cache location, matching the
other backend deployment examples.

For production clusters, if you want a persistent shared Hugging Face cache
instead of per-pod downloads, adapt `agg.yaml` using the guidance in
[Model Caching](../../kubernetes/model-caching.md).

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

By default, FastVideo returns a URL in `data[0].url`. For a self-contained local
smoke test, request inline base64 output explicitly and decode it:

```bash
curl -s -X POST http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prompt": "A cinematic drone shot over a snowy mountain range at sunrise",
    "size": "1920x1088",
    "response_format": "b64_json",
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

If you omit `"response_format": "b64_json"`, the backend uses its default
`"url"` mode and returns `data[0].url`. In local runs without
`--media-output-http-url`, that URL is typically a filesystem-backed path under
`--media-output-fs-url` (default: `file:///tmp/dynamo_media`).

## Worker Configuration Reference

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | HuggingFace model path |
| `--served-model-name` | `--model` | Model name registered with Dynamo discovery |
| `--num-gpus` | `1` | Number of GPUs for distributed inference |
| `--attention-backend` | `TORCH_SDPA` | Sets `FASTVIDEO_ATTENTION_BACKEND`; choices: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `SAGE_SLA_ATTN` |
| `--dit-cpu-offload` / `--no-dit-cpu-offload` | `enabled` | Enable or disable DiT CPU offload |
| `--vae-cpu-offload` / `--no-vae-cpu-offload` | `enabled` | Enable or disable VAE CPU offload |
| `--text-encoder-cpu-offload` / `--no-text-encoder-cpu-offload` | `enabled` | Enable or disable text encoder CPU offload |
| `--torch-compile` / `--no-torch-compile` | `disabled` | Enable or disable `torch.compile` for FastVideo |
| `--torch-compile-mode` | `max-autotune-no-cudagraphs` | `torch.compile` mode to use when compilation is enabled |
| `--torch-compile-fullgraph` / `--no-torch-compile-fullgraph` | `enabled` | Enable or disable fullgraph mode for `torch.compile` |
| `--fp4-quantization` / `--no-fp4-quantization` | `disabled` | Enable FP4 quantization for DiT weights on Blackwell GPUs and newer |
| `--extra-generator-args-file` | — | YAML or JSON file with extra FastVideo generator keyword arguments, including model-specific options such as `ltx2_vae_tiling` |
| `--override-generator-args-json` | — | JSON object string applied on top of the generator args file and built-in defaults |

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
| `FASTVIDEO_ATTENTION_BACKEND` | `TORCH_SDPA` | Attention backend; `dynamo.fastvideo` reads this as the default for `--attention-backend` and then exports the resolved value to FastVideo |
| `FASTVIDEO_STAGE_LOGGING` | `1` | Enable per-stage timing logs |
| `FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT` | `0` | Preserved FastVideo toggle for pre-quantized RMSNorm behavior |
| `FASTVIDEO_LOG_LEVEL` | — | FastVideo-specific log level override; when unset, `dynamo.fastvideo` derives it from `DYN_LOG` |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM during Docker build | Building local runtime bindings or CUDA-dependent packages uses too much RAM | Rebuild on a machine with more RAM |
| 10–20 min wait on first start with `--torch-compile` enabled | Model download + `torch.compile` warmup | Expected behavior; subsequent starts are faster if weights are cached |
| ~35 s second request | Runtime caches still warming | Steady-state performance from third request onward |
| Lower throughput than expected on B200/B300 | `torch.compile`, FP4, and flash-attention are configured separately | Try `--torch-compile --fp4-quantization` and, if desired, `--attention-backend FLASH_ATTN` |
| Startup or import failure after enabling FP4 or changing the attention backend | FP4 and some attention backends depend on specific hardware/software support | Re-run `python -m dynamo.fastvideo` without `--fp4-quantization`, or use `--attention-backend TORCH_SDPA` |

## Source Code

The example source lives at `examples/backends/fastvideo/` in the Dynamo repository.

## See Also

- [vLLM-Omni Text-to-Video](../../backends/vllm/vllm-omni.md#text-to-video) — vLLM-Omni video generation via `/v1/videos`
- [vLLM-Omni Text-to-Image](../../backends/vllm/vllm-omni.md#text-to-image) — vLLM-Omni image generation
- [SGLang Video Generation](../../backends/sglang/sglang-diffusion.md#video-generation) — SGLang video generation worker
- [SGLang Image Diffusion](../../backends/sglang/sglang-diffusion.md#image-diffusion) — SGLang image diffusion worker
- [TRT-LLM Video Diffusion](../../backends/trtllm/trtllm-video-diffusion.md#quick-start) — TensorRT-LLM video diffusion quick start
- [Diffusion Overview](README.md) — Full backend support matrix
