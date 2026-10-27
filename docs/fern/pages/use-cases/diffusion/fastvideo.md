---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: FastVideo
subtitle: Deploy FastVideo text-to-video generation on Dynamo with a custom worker
---

## Why Use FastVideo

[FastVideo](https://github.com/hao-ai-lab/FastVideo) is an open-source video generation framework from Hao AI Lab that cuts generation latency with sparse attention, step distillation, NVFP4 quantization, and fused kernels. In the lab's [real-time 1080p benchmark](https://haoailab.com/blogs/fastvideo_realtime_1080p/), an optimized LTX-2.3 pipeline generates 5 seconds of 1088x1920 video with audio in about 4.55 seconds on a single NVIDIA B200 GPU, which the lab reports as 3.9x lower end-to-end latency than the next-fastest option. Its [Dreamverse](https://haoailab.com/blogs/fastvideo-dreamverse-release/) demo generates a 30-second 1080p clip in about 7 seconds on one B200, and the project reports more than 50x faster denoising from sparse distillation. These are Hao AI Lab figures, not Dynamo benchmarks, and they depend on the model and hardware.

On Dynamo, you run FastVideo as a custom worker behind the `/v1/videos` endpoint, scale it with more workers, and deploy it on Kubernetes. This guide covers that deployment.

> [!NOTE]
> FastVideo is a custom example worker with its own runtime image, not one of Dynamo's built-in backends. Its image encodes H.264 (`libx264`), unlike the standard Dynamo images, which encode VP9. For the built-in backends (vLLM-Omni, SGLang, TensorRT-LLM), see [Text to Video](text-to-video.md).

## Overview

- **Default model:** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`.
- **Typed API:** The worker builds `GeneratorConfig` once at startup and creates a `GenerationRequest` for each `/v1/videos` request.
- **Optimized inference:** `torch.compile` and NVFP4 transformer quantization are available through `--torch-compile` and `--fp4-quantization`; the legacy `--enable-optimizations` flag remains as a shortcut for both.
- **Response format:** Returns one complete MP4 payload per request as `data[0].b64_json` (non-streaming).
- **Concurrency:** One request at a time per worker (VideoGenerator is not re-entrant). Scale throughput by running multiple workers.

> [!IMPORTANT]
> `worker.py` defaults to `--attention-backend VIDEO_SPARSE_ATTN` and routes `VSA_sparsity=0.8` through FastVideo's typed `PipelineSelection.experimental` config. Keep this backend for FastWan 2.1 models; forcing `TORCH_SDPA` can instantiate a non-VSA Wan block and fail checkpoint loading. Use `--attention-backend TORCH_SDPA` for the LTX-2 compatibility path validated by the B300 smoke test.

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

The runtime image is built from the [`Dockerfile`](https://github.com/ai-dynamo/dynamo/blob/main/examples/diffusers/Dockerfile):

- Base image: `nvidia/cuda:13.1.1-devel-ubuntu24.04`
- Installs [FastVideo](https://github.com/hao-ai-lab/FastVideo) `0.2.0` from PyPI, which provides the `fastvideo.api` typed surface and NVFP4 transformer quantization support
- Installs the `ai-dynamo` package with `/v1/videos` support

Build and push the image:

```bash
docker build examples/diffusers/ -t <my-registry/fastvideo-runtime:my-tag>
docker push <my-registry/fastvideo-runtime:my-tag>
```

> [!NOTE]
> A from-source [flash-attention](https://github.com/RandNMR73/flash-attention) (FA4) build is deferred. The worker runs on FastVideo's default attention backends without it.

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

yq '(.spec.components[].podTemplate.spec.containers[] | select(.name == "main") | .image) = env(FASTVIDEO_IMAGE)' \
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

> [!NOTE]
> If this is the first request after startup, expect it to take longer while warmup completes. See [Warmup Time](#warmup-time) for details.

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

### FullHD Video with Audio (LTX-2)

LTX-2 models support native audio generation alongside video. LTX-2 requires width and height divisible by 32, so FullHD requests use `1920x1088` rather than `1920x1080`.

Start the local example with an LTX-2 model:

```bash
cd <dynamo-root>/examples/diffusers/local

MODEL=FastVideo/LTX2-Distilled-Diffusers \
WORKER_EXTRA_ARGS="--attention-backend TORCH_SDPA --fp4-quantization" \
./run_local.sh
```

Send the request from another terminal:

```bash
curl -s -X POST http://localhost:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "FastVideo/LTX2-Distilled-Diffusers",
    "prompt": "A waterfall cascading into a forest pool, birds singing",
    "size": "1920x1088",
    "response_format": "b64_json",
    "nvext": {
      "fps": 24,
      "num_frames": 121,
      "num_inference_steps": 5,
      "guidance_scale": 1.0,
      "seed": 42
    }
  }'
```

FastVideo exposes generated audio and `audio_sample_rate` on its Python result object. This Dynamo worker returns the saved MP4 in `data[0].b64_json`, with FastVideo muxing the generated 24 kHz audio into the MP4 output.

> [!NOTE]
> LTX-2 refine pipeline flags such as `ltx2_refine_enabled`, `ltx2_refine_upsampler_path`, and per-component compile settings are not yet exposed through FastVideo's typed API config.

</Step>

</Steps>

## Warmup Time

On first start, workers download model weights. When `--torch-compile` is enabled, compile/warmup steps can push the first ready time to roughly **10–20 minutes** (hardware-dependent). After the first successful compiled response, the second request can still take around **35 seconds** while runtime caches finish warming up; steady-state performance is typically reached from the third request onward.

> [!TIP]
> The shared Hugging Face cache PVC applied in the [Kubernetes Deployment](#kubernetes-deployment) steps means weights are downloaded once and reused across pod restarts, so warmup is only paid in full on the very first start.

## Worker Configuration Reference

<Tabs>
<Tab title="CLI Flags">

Flags passed to `worker.py`:

<ParamField path="--model, --model-path" type="string" default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers">
  HuggingFace model path.
</ParamField>
<ParamField path="--num-gpus" type="int" default="1">
  Number of GPUs for distributed inference.
</ParamField>
<ParamField path="--discovery-backend" type="string">
  Dynamo discovery backend. Choices: `etcd`, `file`, `mem`, and `kubernetes`. Defaults to the environment, then Kubernetes or file discovery.
</ParamField>
<ParamField path="--attention-backend" type="string" default="VIDEO_SPARSE_ATTN">
  Sets `FASTVIDEO_ATTENTION_BACKEND`. Choices: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `SAGE_ATTN_THREE`, `VIDEO_SPARSE_ATTN`, `VMOBA_ATTN`, `SLA_ATTN`, `SAGE_SLA_ATTN`.
</ParamField>
<ParamField path="--vsa-sparsity" type="float" default="0.8 for VIDEO_SPARSE_ATTN">
  Sets `PipelineSelection.experimental["VSA_sparsity"]`. The worker omits this value for other attention backends unless explicitly set.
</ParamField>
<ParamField path="--torch-compile" type="flag" default="off">
  Enables FastVideo `CompileConfig`.
</ParamField>
<ParamField path="--torch-compile-backend" type="string" default="inductor">
  Sets the `torch.compile` backend.
</ParamField>
<ParamField path="--torch-compile-mode" type="string" default="max-autotune-no-cudagraphs">
  Sets the `torch.compile` mode. Choices: `default`, `reduce-overhead`, `max-autotune`, and `max-autotune-no-cudagraphs`.
</ParamField>
<ParamField path="--torch-compile-fullgraph, --no-torch-compile-fullgraph" type="flag" default="on">
  Controls fullgraph mode.
</ParamField>
<ParamField path="--torch-compile-dynamic, --no-torch-compile-dynamic" type="flag" default="off">
  Controls dynamic shapes.
</ParamField>
<ParamField path="--fp4-quantization" type="flag" default="off">
  Requests NVFP4 transformer quantization through `QuantizationConfig(transformer_quant="NVFP4")`. Confirm activation in worker logs before reporting FP4 results.
</ParamField>
<ParamField path="--enable-optimizations" type="flag" default="off">
  Backward-compatible shortcut for `--torch-compile --fp4-quantization`.
</ParamField>
<ParamField path="--use-fsdp-inference, --no-use-fsdp-inference" type="flag" default="off">
  Controls FastVideo Fully Sharded Data Parallel (FSDP) inference.
</ParamField>
<ParamField path="--dit-cpu-offload, --no-dit-cpu-offload" type="flag" default="on">
  Controls Diffusion Transformer (DiT) CPU offload.
</ParamField>
<ParamField path="--dit-layerwise-offload, --no-dit-layerwise-offload" type="flag" default="on">
  Controls layerwise DiT CPU offload.
</ParamField>
<ParamField path="--vae-cpu-offload, --no-vae-cpu-offload" type="flag" default="on">
  Controls variational autoencoder (VAE) CPU offload.
</ParamField>
<ParamField path="--image-encoder-cpu-offload, --no-image-encoder-cpu-offload" type="flag" default="on">
  Controls image encoder CPU offload.
</ParamField>
<ParamField path="--text-encoder-cpu-offload, --no-text-encoder-cpu-offload" type="flag" default="on">
  Controls text encoder CPU offload.
</ParamField>
<ParamField path="--pin-cpu-memory, --no-pin-cpu-memory" type="flag" default="on">
  Controls pinned host memory for CPU offload transfers.
</ParamField>
<ParamField path="--disable-autocast, --no-disable-autocast" type="flag" default="off">
  Controls autocast in FastVideo denoising and decoding paths.
</ParamField>
<ParamField path="--default-size" type="string" default="1280x720">
  Default request dimensions.
</ParamField>
<ParamField path="--default-seconds" type="int" default="5">
  Default request duration in seconds.
</ParamField>
<ParamField path="--default-fps" type="int" default="24">
  Default request frame rate.
</ParamField>
<ParamField path="--default-num-frames" type="int" default="125">
  Default frame count.
</ParamField>
<ParamField path="--default-num-inference-steps" type="int" default="50">
  Default diffusion inference steps.
</ParamField>
<ParamField path="--default-guidance-scale" type="float" default="1.0">
  Default classifier-free guidance scale.
</ParamField>
<ParamField path="--default-seed" type="int">
  Default random-number-generator seed. When unset, FastVideo uses its preset.
</ParamField>
<ParamField path="--max-video-width" type="int" default="4096">
  Rejects wider requests before calling FastVideo.
</ParamField>
<ParamField path="--max-video-height" type="int" default="4096">
  Rejects taller requests before calling FastVideo.
</ParamField>
<ParamField path="--max-num-frames" type="int" default="1024">
  Rejects requests whose resolved frame count exceeds this cap.
</ParamField>
<ParamField path="--max-num-inference-steps" type="int" default="200">
  Rejects requests whose inference-step count exceeds this cap.
</ParamField>
<ParamField path="--output-dir" type="string" default="$XDG_RUNTIME_DIR/dynamo-fastvideo/outputs or ~/.cache/dynamo/fastvideo/outputs">
  Directory for generated MP4 staging files.
</ParamField>

</Tab>
<Tab title="Request Parameters (nvext)">

Fields nested under `nvext` in the `/v1/videos` request body:

<ParamField path="fps" type="int" default="24">
  Frames per second.
</ParamField>
<ParamField path="num_frames" type="int" default="125">
  Total frames; overrides `fps * seconds` when set.
</ParamField>
<ParamField path="num_inference_steps" type="int" default="50">
  Diffusion inference steps.
</ParamField>
<ParamField path="guidance_scale" type="float" default="1.0">
  Classifier-free guidance scale.
</ParamField>
<ParamField path="seed" type="int">
  Random-number-generator seed override for reproducibility. When unset, FastVideo uses its preset.
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
<ParamField path="FASTVIDEO_ATTENTION_BACKEND" type="string" default="VIDEO_SPARSE_ATTN">
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

### Hardware Support

The worker checks the GPU compute capability at startup and fails fast when the selected attention backend cannot run on the detected hardware:

| Configuration | Minimum compute capability | Notes |
|---|---|---|
| FastWan 2.1 + `VIDEO_SPARSE_ATTN` (default) | 9.0 | `fastvideo-kernel` compiles its Video Sparse Attention (VSA) kernels for sm90a and falls back to a Triton implementation on other architectures. Pre-Hopper GPUs, such as sm86, fail at runtime. FastWan 2.1 checkpoints contain VSA-specific layers, so `TORCH_SDPA` is not a fallback for this model. Validated on B300 (sm103) and RTX 5090 (sm120) |
| LTX-2 + `TORCH_SDPA` | None specific | Compatibility path for GPUs below compute capability 9.0 |
| `--fp4-quantization` (NVFP4) | 10.0 | On older GPUs, the worker logs a warning and continues without NVFP4 quantization |

| Symptom | Cause | Fix |
|---|---|---|
| 10–20 min wait on first start with `--torch-compile` enabled | Model download + `torch.compile` warmup | Expected behavior; subsequent starts are faster if weights are cached |
| ~35 s second request | Runtime caches still warming | Steady-state performance from third request onward |
| Lower throughput than expected on Blackwell GPUs | NVFP4 and compile are opt-in | Pass `--fp4-quantization` and, if needed, `--torch-compile`; confirm NVFP4 activation in worker logs |
| FastWan startup fails with a missing `to_gate_compress` checkpoint parameter | FastWan 2.1 checkpoints expect the VSA Wan block | Use `--attention-backend VIDEO_SPARSE_ATTN --vsa-sparsity 0.8`; do not force `TORCH_SDPA` for FastWan 2.1 |
| Startup or import failure after enabling FP4, compile, or another attention backend | NVFP4 and some attention backends depend on specific hardware and software support | Re-run `worker.py` without `--torch-compile --fp4-quantization`; for LTX-2, use `--attention-backend TORCH_SDPA` |

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

The Compose file builds from the same [`Dockerfile`](https://github.com/ai-dynamo/dynamo/blob/main/examples/diffusers/Dockerfile) used for the Kubernetes runtime image and exposes the API on `http://localhost:8000`.

```bash
cd <dynamo-root>/examples/diffusers/local

# Start 4 workers on GPUs 0..3
COMPOSE_PROFILES=4 docker compose up --build
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
| `MODEL` | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | HuggingFace model path |
| `NUM_GPUS` | `1` | Number of GPUs |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |
| `WORKER_EXTRA_ARGS` | — | Extra flags for `worker.py` (for example, `--attention-backend VIDEO_SPARSE_ATTN --vsa-sparsity 0.8`) |
| `FRONTEND_EXTRA_ARGS` | — | Extra flags for `dynamo.frontend` |

Example:

```bash
MODEL=FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
NUM_GPUS=1 \
DYN_HTTP_PORT=8000 \
WORKER_EXTRA_ARGS="--attention-backend VIDEO_SPARSE_ATTN --vsa-sparsity 0.8" \
./run_local.sh
```

> [!NOTE]
> FastVideo worker flags are not `dynamo.frontend` flags, so pass non-default worker configuration through `WORKER_EXTRA_ARGS`.

Validated B300 smoke-test worker configurations:

| Model | Base worker flags | FP4 worker flags |
|---|---|---|
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | `--attention-backend VIDEO_SPARSE_ATTN --vsa-sparsity 0.8` | `--attention-backend VIDEO_SPARSE_ATTN --vsa-sparsity 0.8 --fp4-quantization` |
| `FastVideo/LTX2-Distilled-Diffusers` | `--attention-backend TORCH_SDPA` | `--attention-backend TORCH_SDPA --fp4-quantization` |

> [!NOTE]
> The FastWan FP4 configuration completed the smoke request, but the worker log did not show the NVFP4 weight-conversion marker. Treat `--fp4-quantization` as requested, not independently confirmed, for FastWan until the FastVideo logs expose that confirmation.

The script writes logs to:

- `.runtime/logs/worker.log`
- `.runtime/logs/frontend.log`

## Source Code

The example source lives at [`examples/diffusers/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers) in the Dynamo repository.

## See Also

- [Text-to-Video with vLLM-Omni](text-to-video.md#vllm-omni) — vLLM-Omni video generation via `/v1/videos`
- [Text-to-Video with SGLang](text-to-video.md#sglang) — SGLang video generation worker
- [Text-to-Video with TensorRT-LLM](text-to-video.md#tensorrt-llm) — TensorRT-LLM diffusion quick start
- [Text to Video](text-to-video.md) — built-in text-to-video backends

