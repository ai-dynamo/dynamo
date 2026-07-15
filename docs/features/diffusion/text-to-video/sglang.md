---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Video with SGLang
subtitle: Generate video from text prompts with SGLang's video generation worker via the /v1/videos endpoint.
sidebar-title: SGLang
---

Video generation workers produce videos from text prompts using SGLang's `DiffGenerator` with frame-to-video encoding, via the `--video-generation-worker` flag. The same worker also supports [image-to-video](../image-to-video/sglang.md). See the [Diffusion Overview](../README.md) for backend setup.

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

- [Image-to-Video with SGLang](../image-to-video/sglang.md)
- [Text-to-Video with vLLM-Omni](vllm-omni.md)
- [Text-to-Video with TensorRT-LLM](trtllm.md)
- [SGLang Examples](../../../backends/sglang/sglang-examples.md)
