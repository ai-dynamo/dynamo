---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM-Omni
---

# [Experimental] Omni Models with vLLM

Dynamo supports multimodal generation through the [vLLM-Omni](https://github.com/vllm-project/vllm-omni) backend. This integration exposes text-to-text, text-to-image, and text-to-video capabilities via OpenAI-compatible API endpoints.

## Prerequisites

This guide assumes familiarity with deploying Dynamo with vLLM as described in the [vLLM backend guide](/docs/pages/backends/vllm/README.md).

## Supported Modalities

| Modality | Endpoint(s) | `--output-modalities` |
|---|---|---|
| Text-to-Text | `/v1/chat/completions` | `text` (default) |
| Text-to-Image | `/v1/chat/completions`, `/v1/images/generations` | `image` |
| Text-to-Video | `/v1/videos` | `video` |

The `--output-modalities` flag determines which endpoint(s) the worker registers. When set to `image`, both `/v1/chat/completions` (returns inline base64 images) and `/v1/images/generations` are available. When set to `video`, the worker serves `/v1/videos`.

## Text-to-Text

Launch an aggregated deployment (frontend + omni worker):

```bash
bash examples/backends/vllm/launch/agg_omni.sh
```

This starts `Qwen/Qwen2.5-Omni-7B` with a single-stage thinker config on one GPU.

Verify the deployment:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "stream": false
  }'
```

This script uses a custom stage config (`stage_configs/single_stage_llm.yaml`) that configures the thinker stage for text generation. See [Stage Configuration](#stage-configuration) for details.

## Text-to-Image

Launch using the provided script with `Qwen-Image`:

```bash
bash examples/backends/vllm/launch/agg_omni_image.sh
```

### Via `/v1/chat/completions`

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen-Image",
    "messages": [{"role": "user", "content": "A cat sitting on a windowsill"}],
    "stream": false
  }'
```

The response includes base64-encoded images inline:

```json
{
  "choices": [{
    "delta": {
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  }]
}
```

### Via `/v1/images/generations`

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen-Image",
    "prompt": "A cat sitting on a windowsill",
    "size": "1024x1024",
    "response_format": "url"
  }'
```

## Text-to-Video

Launch using the provided script with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh
```

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

The response returns a video URL or base64 data depending on `response_format`:

```json
{
  "id": "...",
  "object": "video",
  "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "status": "completed",
  "data": [{"url": "/tmp/generated_video.mp4"}]
}
```

The `/v1/videos` endpoint also accepts NVIDIA extensions via the `nvext` field for fine-grained control:

| Field | Description | Default |
|---|---|---|
| `nvext.fps` | Frames per second | 24 |
| `nvext.num_frames` | Number of frames (overrides `fps * seconds`) | -- |
| `nvext.negative_prompt` | Negative prompt for guidance | -- |
| `nvext.num_inference_steps` | Number of denoising steps | 50 |
| `nvext.guidance_scale` | CFG guidance scale | 5.0 |
| `nvext.seed` | Random seed for reproducibility | -- |

## CLI Reference

| Flag | Description |
|---|---|
| `--omni` | Enable the vLLM-Omni orchestrator (required for all omni workloads) |
| `--output-modalities <modality>` | Output modality: `text`, `image`, `video`, or `audio` |
| `--stage-configs-path <path>` | Path to stage config YAML (optional; vLLM-Omni uses model defaults if omitted) |
| `--connector none` | Disable KV connector (recommended for omni workers) |

## Stage Configuration

Omni pipelines are configured via YAML stage configs. See [`examples/backends/vllm/launch/stage_configs/single_stage_llm.yaml`](/examples/backends/vllm/launch/stage_configs/single_stage_llm.yaml) for an example. For full documentation on stage config format and multi-stage pipelines, refer to the [vLLM-Omni Stage Configs documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/).

## Current Limitations

- Only text prompts are supported as input (no multimodal input yet).
- KV cache events are not published for omni workers.
- Each worker supports a single output modality at a time.
