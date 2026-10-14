---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Video
subtitle: Generate videos from text prompts with vLLM-Omni, SGLang, or TensorRT-LLM
---

Dynamo serves text-to-video models through the `/v1/videos` endpoint. To
animate a source image, see [Image-to-Video](image-to-video.md). To generate
video with a synchronized audio track, see [MiniMax-H3](minimax-h3.md). To
answer questions *about* a video instead, see [Video Understanding](video-understanding.md).

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

## Generate Videos

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Text-to-video generation runs a vLLM-Omni worker with `--output-modalities video`.

### Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default model (1 GPU) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | |
| `MiniMaxAI/MiniMax-H3` | Joint 24-FPS video and 32-kHz stereo audio; requires the dedicated [MiniMax-H3 workflow](minimax-h3.md) and 4 B200 GPUs |

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
- [Text-to-Image with TensorRT-LLM](../image/image-generation.md#tensorrt-llm)

</Tab>
</Tabs>
