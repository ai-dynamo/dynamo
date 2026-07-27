---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Video
subtitle: Generate videos from text prompts with vLLM-Omni, SGLang, TensorRT-LLM, or FastVideo
---

Choose a backend for text-to-video generation. See the [Diffusion Overview](../README.md) for installation and shared configuration.

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Text-to-video generation runs a vLLM-Omni worker with `--output-modalities video`.

## Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default model (1 GPU) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

## Launch

Launch using the provided script with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh
```

## Generate a Video

Generate a video via `/v1/videos`:

```bash
curl -s http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prompt": "A drone flyover of a mountain landscape",
    "seconds": 2,
    "size": "832x480",
    "response_format": "url"
  }'
```

The response returns a video URL or base64 data depending on `response_format` (e.g. `{"object": "video", "status": "completed", "data": [{"url": "file:///tmp/dynamo_media/videos/req-abc123.mp4"}]}`).

## Request Parameters (`nvext`)

The `/v1/videos` endpoint also accepts NVIDIA extensions via the `nvext` field for fine-grained control:

<ParamField path="nvext.fps" type="int" default="24">
  Frames per second.
</ParamField>
<ParamField path="nvext.num_frames" type="int">
  Number of frames (overrides `fps * seconds`).
</ParamField>
<ParamField path="nvext.negative_prompt" type="string">
  Negative prompt for guidance.
</ParamField>
<ParamField path="nvext.num_inference_steps" type="int" default="50">
  Number of denoising steps.
</ParamField>
<ParamField path="nvext.guidance_scale" type="float" default="5.0">
  CFG guidance scale.
</ParamField>
<ParamField path="nvext.seed" type="int">
  Random seed for reproducibility.
</ParamField>

<Note>
The `nvext.boundary_ratio` and `nvext.guidance_scale_2` fields apply to the dual-expert MoE schedule used in image-to-video. See [Image-to-Video with vLLM-Omni](../image-to-video/README.md#vllm-omni).
</Note>

## See Also

- [Image-to-Video with vLLM-Omni](../image-to-video/README.md#vllm-omni) — animate a source image with the same `/v1/videos` endpoint
- [Text-to-Video with SGLang](README.md#sglang)
- [Text-to-Video with TensorRT-LLM](README.md#tensorrt-llm)
- [Text-to-Video with FastVideo](README.md#fastvideo)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)

</Tab>
<Tab title="SGLang">

<Anchor id="sglang" />

Video generation workers produce videos from text prompts using SGLang's `DiffGenerator` with frame-to-video encoding, via the `--video-generation-worker` flag. The same worker also supports [image-to-video](../image-to-video/README.md#sglang).

## Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/text-to-video-diffusion.sh
```

Use `--wan-size 1b` (default, 1 GPU) or `--wan-size 14b` (2 GPUs). See the launch script for all configuration options.

## Generate a Video

```bash
curl http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Roger Federer winning his 19th grand slam",
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "seconds": 2,
    "size": "832x480",
    "response_format": "url",
    "nvext": {
      "fps": 8,
      "num_frames": 17,
      "num_inference_steps": 50
    }
  }'
```

## See Also

- [Image-to-Video with SGLang](../image-to-video/README.md#sglang)
- [Text-to-Video with vLLM-Omni](README.md#vllm-omni)
- [Text-to-Video with TensorRT-LLM](README.md#tensorrt-llm)
- [SGLang Examples](../../../backends/sglang/sglang-examples.mdx)

</Tab>
<Tab title="TensorRT-LLM">

<Anchor id="tensorrt-llm" />

TensorRT-LLM supports **experimental** text-to-video generation through the `--modality video_diffusion` flag. See the [Diffusion Overview](../README.md) for requirements and installation (including the ffmpeg/imageio setup needed for MP4 encoding).

## Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

## Launch

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --media-output-fs-url file:///tmp/dynamo_media
```

## Generate a Video

Video generation uses the `/v1/videos` endpoint:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "model": "wan_t2v",
    "seconds": 4,
    "size": "832x480",
    "nvext": {
      "fps": 24
    }
  }'
```

## Configuration

For the full flag surface (quantization, TeaCache, torch.compile, attention backend, and request defaults), see the [TensorRT-LLM Configuration reference](../../../backends/trtllm/trtllm-config-reference.mdx#diffusion-experimental).

## Limitations

- Diffusion is experimental and not recommended for production use.
- Only text-to-video and text-to-image are supported in this release (image-to-video planned).
- Requires a GPU with sufficient VRAM for the diffusion model.

## See Also

- [Text-to-Video with vLLM-Omni](README.md#vllm-omni)
- [Text-to-Video with SGLang](README.md#sglang)
- [Text-to-Video with FastVideo](README.md#fastvideo)
- [Text-to-Image with TensorRT-LLM](../text-to-image/README.md#tensorrt-llm)

</Tab>
<Tab title="FastVideo">

<Anchor id="fastvideo" />

This guide covers deploying [FastVideo](https://github.com/hao-ai-lab/FastVideo) text-to-video generation on Dynamo using a custom worker (`worker.py`) exposed through the `/v1/videos` endpoint.

<Note>
Dynamo also supports text-to-video through built-in backends: [vLLM-Omni](README.md#vllm-omni), [SGLang](README.md#sglang), and [TensorRT-LLM](README.md#tensorrt-llm). See the [Diffusion Overview](../README.md) for the full support matrix across all modalities.
</Note>

## Overview

- **Default model:** `FastVideo/LTX2-Distilled-Diffusers` — a distilled variant of the LTX-2 Diffusion Transformer (Lightricks), reducing inference from 50+ steps to just 5.
- **Two-stage pipeline:** Stage 1 generates video at target resolution; Stage 2 refines with a distilled LoRA for improved fidelity and texture.
- **Optimized inference:** FP4 quantization and `torch.compile` are available via `--enable-optimizations`; attention backend selection is controlled separately via `--attention-backend`.
- **Response format:** Returns one complete MP4 payload per request as `data[0].b64_json` (non-streaming).
- **Concurrency:** One request at a time per worker (VideoGenerator is not re-entrant). Scale throughput by running multiple workers.

<Warning>
`worker.py` defaults to `--attention-backend TORCH_SDPA` for broader compatibility across GPUs, including systems such as H100. For the B200/B300-oriented path, enable FP4/compile with `--enable-optimizations` and, if desired, opt into flash-attention explicitly with `--attention-backend FLASH_ATTN`.
</Warning>

## Kubernetes Deployment

Kubernetes is the recommended path for running FastVideo on Dynamo. The steps below build and push the runtime image, deploy the aggregated worker, and send a first request. For a single-node development workflow, see [Local Deployment](#local-deployment) at the bottom of this page.

### Deployment Files

| File | Description |
|---|---|
| `agg.yaml` | Base aggregated deployment (Frontend + `FastVideoWorker`) |
| `agg_user_workload.yaml` | Same deployment with `user-workload` tolerations and `imagePullSecrets` |
| `huggingface-cache-pvc.yaml` | Shared HF cache PVC for model weights |
| `dynamo-platform-values-user-workload.yaml` | Optional Helm values for clusters with tainted `user-workload` nodes |

### Prerequisites

- Dynamo Kubernetes Platform installed
- GPU-enabled Kubernetes cluster
- A container registry you can push the FastVideo runtime image to
- Optional HF token secret (for gated models)

<Steps>

<Step title="Build and push the FastVideo runtime image">

The runtime image is built from the [`Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile):

- Base image: `nvidia/cuda:13.1.1-devel-ubuntu24.04`
- Installs [FastVideo](https://github.com/hao-ai-lab/FastVideo) from GitHub
- Installs Dynamo from the `release/1.0.0` branch (for `/v1/videos` support)
- Compiles a [flash-attention](https://github.com/RandNMR73/flash-attention) fork from source

The Dockerfile exposes `TORCH_CUDA_ARCH_LIST` as a build argument (default: `10.0 10.0a` for Blackwell). Pass `--build-arg` to target a different architecture, and use `MAX_JOBS` (default: `4`) to bound the parallel flash-attention compile jobs:

```bash
# Blackwell (default)
docker build examples/diffusers/ --build-arg TORCH_CUDA_ARCH_LIST="10.0 10.0a" -t <my-registry/fastvideo-runtime:my-tag>

# Hopper, on a memory-constrained builder
docker build examples/diffusers/ \
  --build-arg TORCH_CUDA_ARCH_LIST="9.0 9.0a" \
  --build-arg MAX_JOBS=2 \
  -t <my-registry/fastvideo-runtime:my-tag>

docker push <my-registry/fastvideo-runtime:my-tag>
```

<Warning>
The first image build can take **20–40+ minutes** because FastVideo and CUDA-dependent components are compiled during the build. Subsequent builds are much faster if Docker layer cache is preserved. Compiling `flash-attention` can use significant RAM — low-memory builders may hit out-of-memory failures. If that happens, lower `MAX_JOBS`. The [flash-attn install notes](https://pypi.org/project/flash-attn/) specifically recommend this on machines with less than 96 GB RAM and many CPU cores.
</Warning>

</Step>

<Step title="Create the Hugging Face token secret (optional)">

Only required for gated models:

```bash
export NAMESPACE=<your-namespace>
export HF_TOKEN=<your-hf-token>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

</Step>

<Step title="Apply the cache PVC and deployment">

Mounting a shared Hugging Face cache PVC means model weights are downloaded once and reused across pod restarts.

```bash
cd <dynamo-root>/examples/diffusers/deploy
export NAMESPACE=<your-namespace>

kubectl apply -f huggingface-cache-pvc.yaml -n ${NAMESPACE}
kubectl apply -f agg.yaml -n ${NAMESPACE}
```

For clusters with tainted `user-workload` nodes and private registry pulls, set your pull secret name and image in `agg_user_workload.yaml`, then apply that variant instead:

```bash
kubectl apply -f huggingface-cache-pvc.yaml -n ${NAMESPACE}
kubectl apply -f agg_user_workload.yaml -n ${NAMESPACE}
```

To swap the image on an existing deployment quickly:

```bash
export DEPLOYMENT_FILE=agg.yaml
export FASTVIDEO_IMAGE=<my-registry/fastvideo-runtime:my-tag>

yq '.spec.services.[].extraPodSpec.mainContainer.image = env(FASTVIDEO_IMAGE)' \
  ${DEPLOYMENT_FILE} > ${DEPLOYMENT_FILE}.generated

kubectl apply -f ${DEPLOYMENT_FILE}.generated -n ${NAMESPACE}
```

</Step>

<Step title="Verify and access the deployment">

```bash
kubectl get dgd -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE}
kubectl logs -n ${NAMESPACE} -l nvidia.com/dynamo-component=FastVideoWorker
```

Port-forward the Frontend Service:

```bash
kubectl port-forward -n ${NAMESPACE} svc/fastvideo-agg-frontend 8000:8000
```

</Step>

<Step title="Send a test request">

<Note>
If this is the first request after startup, expect it to take longer while warmup completes. See [Warmup Time](#warmup-time) for details.
</Note>

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

</Step>

</Steps>

## Warmup Time

On first start, workers download model weights. When `--enable-optimizations` is enabled, compile/warmup steps can push the first ready time to roughly **10–20 minutes** (hardware-dependent). After the first successful optimized response, the second request can still take around **35 seconds** while runtime caches finish warming up; steady-state performance is typically reached from the third request onward.

<Tip>
The shared Hugging Face cache PVC applied in the [Kubernetes Deployment](#kubernetes-deployment) steps means weights are downloaded once and reused across pod restarts, so warmup is only paid in full on the very first start.
</Tip>

## Worker Configuration Reference

<Tabs>
<Tab title="CLI Flags">

Flags passed to `worker.py`:

<ParamField path="--model" type="string" default="FastVideo/LTX2-Distilled-Diffusers">
  HuggingFace model path.
</ParamField>
<ParamField path="--num-gpus" type="int" default="1">
  Number of GPUs for distributed inference.
</ParamField>
<ParamField path="--enable-optimizations" type="flag" default="off">
  Enables FP4 quantization and `torch.compile`.
</ParamField>
<ParamField path="--attention-backend" type="string" default="TORCH_SDPA">
  Sets `FASTVIDEO_ATTENTION_BACKEND`. Choices: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `SAGE_SLA_ATTN`.
</ParamField>

</Tab>
<Tab title="Request Parameters (nvext)">

Fields nested under `nvext` in the `/v1/videos` request body:

<ParamField path="fps" type="int" default="24">
  Frames per second.
</ParamField>
<ParamField path="num_frames" type="int" default="121">
  Total frames; overrides `fps * seconds` when set.
</ParamField>
<ParamField path="num_inference_steps" type="int" default="5">
  Diffusion inference steps.
</ParamField>
<ParamField path="guidance_scale" type="float" default="1.0">
  Classifier-free guidance scale.
</ParamField>
<ParamField path="seed" type="int" default="10">
  RNG seed for reproducibility.
</ParamField>
<ParamField path="negative_prompt" type="string">
  Text to avoid in generation.
</ParamField>

</Tab>
<Tab title="Environment Variables">

<ParamField path="FASTVIDEO_VIDEO_CODEC" type="string" default="libx264">
  Video codec for MP4 encoding.
</ParamField>
<ParamField path="FASTVIDEO_X264_PRESET" type="string" default="ultrafast">
  x264 encoding speed preset.
</ParamField>
<ParamField path="FASTVIDEO_ATTENTION_BACKEND" type="string" default="TORCH_SDPA">
  Attention backend; `worker.py` sets this from `--attention-backend` and validates `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, and `SAGE_SLA_ATTN`.
</ParamField>
<ParamField path="FASTVIDEO_STAGE_LOGGING" type="string" default="1">
  Enable per-stage timing logs.
</ParamField>
<ParamField path="FASTVIDEO_LOG_LEVEL" type="string">
  Set to `DEBUG` for verbose logging.
</ParamField>

</Tab>
</Tabs>

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM during Docker build | `flash-attention` compilation uses too much RAM | Pass `--build-arg MAX_JOBS=2` (or lower) at build time |
| `no kernel image available for this GPU` or CUDA arch error at runtime | Image was built for a different GPU architecture | Rebuild with the correct `TORCH_CUDA_ARCH_LIST` (e.g. `9.0 9.0a` for Hopper) |
| 10–20 min wait on first start with optimizations enabled | Model download + `torch.compile` warmup | Expected behavior; subsequent starts are faster if weights are cached |
| ~35 s second request | Runtime caches still warming | Steady-state performance from third request onward |
| Lower throughput than expected on B200/B300 | FP4/compile and flash-attention are configured separately | Pass `--enable-optimizations` and, if desired, `--attention-backend FLASH_ATTN` |
| Startup or import failure after enabling optimizations or changing the attention backend | FP4 and some attention backends depend on specific hardware/software support | Re-run `worker.py` without `--enable-optimizations`, or use `--attention-backend TORCH_SDPA` |

## Local Deployment

For single-node development you can run FastVideo directly with Docker Compose or a host-local script instead of Kubernetes.

### Prerequisites

**For Docker Compose:**

- Docker Engine 26.0+
- Docker Compose v2
- NVIDIA Container Toolkit

**For host-local script:**

- Python environment with Dynamo + FastVideo dependencies installed
- CUDA-compatible GPU runtime available on host

### Option 1: Docker Compose

The Compose file builds from the same [`Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile) used for the Kubernetes runtime image and exposes the API on `http://localhost:8000`. Set `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` as environment variables to control the build (see the build notes in the [Kubernetes Deployment](#kubernetes-deployment) steps for time and memory expectations).

```bash
cd <dynamo-root>/examples/diffusers/local

# Start 4 workers on GPUs 0..3
COMPOSE_PROFILES=4 docker compose up --build

# Hopper on a memory-constrained builder
TORCH_CUDA_ARCH_LIST="9.0 9.0a" MAX_JOBS=2 COMPOSE_PROFILES=4 docker compose up --build
```

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
| `WORKER_EXTRA_ARGS` | — | Extra flags for `worker.py` (for example, `--enable-optimizations --attention-backend FLASH_ATTN`) |
| `FRONTEND_EXTRA_ARGS` | — | Extra flags for `dynamo.frontend` |

Example:

```bash
MODEL=FastVideo/LTX2-Distilled-Diffusers \
NUM_GPUS=1 \
HTTP_PORT=8000 \
WORKER_EXTRA_ARGS="--enable-optimizations --attention-backend FLASH_ATTN" \
./run_local.sh
```

<Note>
`--enable-optimizations` and `--attention-backend` are `worker.py` flags, not `dynamo.frontend` flags, so pass them through `WORKER_EXTRA_ARGS` when you want a non-default worker configuration.
</Note>

The script writes logs to:

- `.runtime/logs/worker.log`
- `.runtime/logs/frontend.log`

## Source Code

The example source lives at [`examples/diffusers/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers) in the Dynamo repository.

## See Also

- [Text-to-Video with vLLM-Omni](README.md#vllm-omni) — vLLM-Omni video generation via `/v1/videos`
- [Text-to-Video with SGLang](README.md#sglang) — SGLang video generation worker
- [Text-to-Video with TensorRT-LLM](README.md#tensorrt-llm) — TensorRT-LLM diffusion quick start
- [Diffusion Overview](../README.md) — Full backend support matrix

</Tab>
</Tabs>
