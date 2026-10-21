---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text to Video (T2V)
subtitle: Generate video from text prompts, with or without a synchronized audio track
---

> [!TIP]
> To serve FastVideo models, see [FastVideo](fastvideo.md). It runs a custom
> worker behind the same `/v1/videos` endpoint and has its own Kubernetes deployment path.

Dynamo serves text-to-video models through the `/v1/videos` endpoint. To
animate a source image, see [Image to Video (I2V)](image-to-video.md). To answer
questions *about* a video instead, see [Video Understanding](../video-understanding/overview.md).

## Tutorials on This Page

| Tutorial | Output | Backends | Hardware |
| --- | --- | --- | --- |
| [MiniMax-H3: video with audio](#minimax-h3-video-with-audio) | Video with a synchronized audio track (T2VA), H.264 and AAC | vLLM-Omni with a video-audio image overlay | 4× B200 |
| [Text to video](#text-to-video) | Video only, VP9 | vLLM-Omni, SGLang, or TensorRT-LLM (experimental) | One GPU for the default models |

Both tutorials use the backend setup in [Set up the backend](#set-up-the-backend).

> [!IMPORTANT]
> **Generated video is VP9.** Dynamo's runtime images encode video output as
> VP9, including `output_format: "mp4"` (VP9 in an MP4 container). Earlier
> releases returned H.264 for MP4 output. VP9-in-MP4 plays in Chrome, Firefox,
> and ffmpeg, but not in Safari or QuickTime. To get H.264, transcode on the
> client: `ffmpeg -i output.mp4 -c:v libx264 output-h264.mp4`. Dynamo uses a CPU
> VP9 encoder because A100, H100, H200, B200, and GB200 have no hardware video
> encoder (NVENC).

> [!NOTE]
> Video generation backends run from Dynamo CLI launch scripts and
> `python -m dynamo.*` entrypoints. Dynamo does not yet ship Kubernetes recipes
> for them. [FastVideo](fastvideo.md), a custom example worker, has its own
> Kubernetes deployment path.

## MiniMax-H3: Video with Audio

MiniMax-H3 generates video with a synchronized audio track (text-to-video-and-audio,
T2VA). It runs on vLLM-Omni and needs 4× B200 GPUs. Because the standard Dynamo
image encodes VP9 video only, MiniMax-H3 also needs a separate image layer that
muxes H.264 video and AAC audio. Set up vLLM-Omni first, as described in
[Set up the backend](#set-up-the-backend).

The initial MiniMax-H3 qualification serves text-to-video-and-audio (T2VA) with
one aggregated diffusion worker. The launcher passes `--task-type fl2va`, which
loads only H3's FL2VA checkpoint partition. Dynamo's standard image stays on
its VP9-only media stack, so build the opt-in video-audio overlay to mux H.264
video and AAC audio:

```bash
export DYN_H3_DYNAMO_REVISION="$(git rev-parse HEAD)"
export DYN_H3_BASE_IMAGE="dynamo:${DYN_H3_DYNAMO_REVISION}-vllm-runtime"
export DYN_H3_IMAGE="dynamo:minimax-h3-${DYN_H3_DYNAMO_REVISION}"

python container/render.py --framework vllm --target runtime --output-short-filename
docker build -t "$DYN_H3_BASE_IMAGE" -f container/rendered.Dockerfile .
docker build \
  --build-arg BASE_IMAGE="$DYN_H3_BASE_IMAGE" \
  -f examples/backends/vllm/omni/video_audio.Dockerfile \
  -t "$DYN_H3_IMAGE" .
export DYN_H3_IMAGE_REF="$(docker image inspect --format '{{.Id}}' "$DYN_H3_IMAGE")"
```

Start a shell with exactly four B200 GPUs visible. The commands pass the tested
source revision and image ID into the qualification record:

```bash
container/run.sh \
  --image "$DYN_H3_IMAGE" \
  --gpus device=0,1,2,3 \
  --mount-workspace \
  -e DYN_H3_DYNAMO_REVISION \
  -e DYN_H3_IMAGE_REF \
  -e DYN_H3_MODEL \
  -e DYN_H3_MODEL_PATH \
  -e DYN_H3_MODEL_REVISION \
  -e DYN_H3_FASTH3_LORA_PATH \
  -e DYN_H3_FASTH3_VARIANT \
  -e DYN_H3_FASTVIDEO_VSA_TOPK \
  -e HF_TOKEN \
  -it
```

To use a FastH3 adapter, export `DYN_H3_FASTH3_LORA_PATH` (and optionally `DYN_H3_FASTH3_VARIANT`) as described below **before** you run the launcher. The launcher reads them at startup, so stop and relaunch the worker to change adapters or variants.

Launch the profile inside the container. It uses four-way Ulysses and text-encoder tensor
parallelism while keeping VAE patch parallelism at one, so small supported
resolutions such as 448x256 still have at least one tile per participating
rank. Larger resolutions can opt into four-way VAE patch parallelism with
`DYN_H3_VAE_PATCH_PARALLEL_SIZE=4`:

```bash
bash examples/backends/vllm/launch/agg_omni_minimax_h3.sh \
  > /tmp/dynamo-minimax-h3.log 2>&1 &
```

When an offline cache contains only H3's `FL2VA/` partition, set
`DYN_H3_MODEL_PATH` to that partition directory. The launcher keeps
`DYN_H3_MODEL` as the public API model name and passes the local partition as
the engine path.

To use the four-step Dense FastH3 adapter, download and pin the startup adapter before you launch the worker:

```bash
export DYN_H3_FASTH3_LORA_PATH="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision bcf40ca6f457ed66f8badf13514943e390205fca)"
```

The launcher fuses this adapter into the checkpoint at load time. It is not a
request-time LoRA and cannot be combined with another request-time adapter.
The qualification script verifies the adapter SHA-256, requests four inference
steps, omits request scheduler-shift overrides, and records request duration and
adapter provenance. FastH3 v1 is T2VA-only and does not support CPU or layerwise
offload.

To use FastH3 with Variable Sparse Attention (VSA), download a VSA adapter and
identify the variant to the launcher and qualification script:

```bash
export DYN_H3_FASTH3_VARIANT=vsa-datafree
export DYN_H3_FASTH3_LORA_PATH="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  vsa-datafree/adapter_model.safetensors \
  --revision bcf40ca6f457ed66f8badf13514943e390205fca)"
```

This path requires vLLM-Omni `v0.29.0rc1` or later. The video-audio overlay
pins `fastvideo-kernel==0.3.5`. For a `vsa-*` adapter, the launcher selects
`FASTVIDEO_VSA` and uses the portable Triton route, retaining 64 key/value
blocks per query block. Override the top-k value with
`DYN_H3_FASTVIDEO_VSA_TOPK`. FastH3 VSA supports pure Ulysses sequence
parallelism; ring and all-gather degrees must remain 1. Check the first-forward
worker logs for `FASTVIDEO_VSA H3 routing` and verify that it does not fall back
to dense SDPA.

Run the supplied end-to-end qualification against the same worker. It requests
a 10-second clip, then verifies a non-empty H.264 stream at 24 FPS and a
non-silent 32-kHz stereo AAC stream:

```bash
export DYN_H3_QUAL_DIR=/tmp/dynamo_minimax_h3_qualification
bash examples/backends/vllm/launch/validate_omni_minimax_h3.sh
```

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "A cat playing Canon in D on a grand piano.",
  "size": "448x256",
  "response_format": "url",
  "output_format": "mp4",
  "nvext": {
    "fps": 24,
    "num_inference_steps": 50,
    "seed": 42
  },
  "task": "t2va",
  "duration": 10.0,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "aspect_ratio": "16:9"
}
```

The launcher pins model revision `42ed227ee7df40d41602854ae760620d6eb651fe`.
Set `DYN_H3_MODEL_REVISION` to test another revision. Dynamo preserves
unrecognized top-level video fields and the vLLM-Omni adapter merges them into
the pipeline's `extra_args`, where the upstream model validates them. Client
libraries that expose an `extra_body` option can use it to add these top-level
fields; do not send a literal nested `extra_body` object. Generated responses
report `fps` and `audio_sample_rate` for each MP4. The qualification output
records the exact Dynamo revision, container image ID, model revision, package
versions, visible GPU models, request, media checksum, and stream probe data.

First/last-frame FL2VA and reference-driven Ref2VA are not part of this initial
qualification. They additionally require typed image, video, and audio input
references.

## Which Backend?

| Backend | Status | Text-to-video | Image-to-video |
| --- | --- | --- | --- |
| vLLM-Omni | Supported | Yes | Yes |
| SGLang | Supported | Yes | Yes |
| TensorRT-LLM | Experimental | Yes | No |
| [FastVideo](fastvideo.md) (custom worker) | Supported | Yes | No |

## Set up the backend

<Tabs>
<Tab title="vLLM-Omni">

<Markdown src="../_snippets/setup-vllm-omni.mdx" />

See the [vLLM backend setup](../../developer-guide/knowledge-base/modular-components/backends/vllm/overview.md) and the [vLLM-Omni Configuration reference](../../reference/backends/vllm-omni-configuration.mdx) for every flag.

</Tab>
<Tab title="SGLang">

<Markdown src="../_snippets/setup-sglang-generation.mdx" />

See the [SGLang backend setup](../../developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) and the [SGLang Configuration reference](../../reference/backends/sglang-configuration.mdx) for every flag.

</Tab>
<Tab title="TensorRT-LLM">

<Markdown src="../_snippets/setup-trtllm-visual-gen.mdx" />

See the [TensorRT-LLM Configuration reference](../../reference/backends/tensorrt-llm-configuration.mdx) for every flag.

</Tab>
</Tabs>

## Text to Video

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Text-to-video generation runs a vLLM-Omni worker with `--output-modalities video`.

### Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default model (1 GPU) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | |
| `MiniMaxAI/MiniMax-H3` | Joint 24-FPS video and 32-kHz stereo audio; requires the dedicated [MiniMax-H3 workflow](#minimax-h3-video-with-audio) and 4 B200 GPUs |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

### Launch

Launch using the provided script with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh
```

### Generate a Video

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

### Request Parameters (`nvext`)

The `/v1/videos` endpoint also accepts NVIDIA extensions via the `nvext` field for fine-grained control:

<ParamField path="nvext.fps" type="int" default="16">
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

> [!NOTE]
> The `nvext.boundary_ratio` and `nvext.guidance_scale_2` fields apply to the dual-expert MoE schedule used in image-to-video. See [Image-to-Video with vLLM-Omni](image-to-video.md#vllm-omni).

### See Also

- [Image-to-Video with vLLM-Omni](image-to-video.md#vllm-omni) — animate a source image with the same `/v1/videos` endpoint
- [Text-to-Video with SGLang](text-to-video.md#sglang)
- [Text-to-Video with TensorRT-LLM](text-to-video.md#tensorrt-llm)
- [FastVideo](fastvideo.md)
- [vLLM-Omni Configuration reference](../../reference/backends/vllm-omni-configuration.mdx)

</Tab>
<Tab title="SGLang">

<Anchor id="sglang" />

Video generation workers produce videos from text prompts using SGLang's `DiffGenerator` with frame-to-video encoding, via the `--video-generation-worker` flag. The same worker also supports [image-to-video](image-to-video.md#sglang).

### Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/text-to-video-diffusion.sh
```

Use `--wan-size 1b` (default, 1 GPU) or `--wan-size 14b` (2 GPUs). See the launch script for all configuration options.

### Generate a Video

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

### See Also

- [Image-to-Video with SGLang](image-to-video.md#sglang)
- [Text-to-Video with vLLM-Omni](text-to-video.md#vllm-omni)
- [Text-to-Video with TensorRT-LLM](text-to-video.md#tensorrt-llm)
- [SGLang Examples](../../recipes/cli-templates/sglang.mdx)

</Tab>
<Tab title="TensorRT-LLM">

<Anchor id="tensorrt-llm" />

TensorRT-LLM supports **experimental** text-to-video generation through the `--modality video_diffusion` flag. Set it up as described in [Set up the backend](#set-up-the-backend), including the ffmpeg/imageio step for MP4 encoding.

### Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

### Launch

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --served-model-name wan_t2v \
  --media-output-fs-url file:///tmp/dynamo_media
```

### Generate a Video

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

### Configuration

For the full flag surface (quantization, TeaCache, torch.compile, attention backend, and request defaults), see the [TensorRT-LLM Configuration reference](../../reference/backends/tensorrt-llm-configuration.mdx#diffusion-experimental).

### Limitations

- Diffusion is experimental and not recommended for production use.
- Only text-to-video and text-to-image are supported in this release (image-to-video planned).
- Requires a GPU with sufficient VRAM for the diffusion model.

### See Also

- [Text-to-Video with vLLM-Omni](text-to-video.md#vllm-omni)
- [Text-to-Video with SGLang](text-to-video.md#sglang)
- [FastVideo](fastvideo.md)
- [Text-to-Image with TensorRT-LLM](text-to-image.md#tensorrt-llm)

</Tab>
</Tabs>
