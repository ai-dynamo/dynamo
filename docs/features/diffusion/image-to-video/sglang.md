---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Image-to-Video with SGLang
subtitle: Animate a source image with SGLang's video generation worker via the /v1/videos endpoint.
sidebar-title: SGLang
---

SGLang's video generation worker (`--video-generation-worker`) supports image-to-video (I2V) in addition to [text-to-video](../text-to-video/sglang.md), using SGLang's `DiffGenerator` with frame-to-video encoding. See the [Diffusion Overview](../README.md) for backend setup.

## Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/text-to-video-diffusion.sh
```

Use `--wan-size 1b` (default, 1 GPU) or `--wan-size 14b` (2 GPUs). See the launch script for all configuration options.

## Generate a Video from an Image

Provide the source image alongside the prompt in the `/v1/videos` request. See the [launch script](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/launch/text-to-video-diffusion.sh) and [SGLang Examples](../../../backends/sglang/sglang-examples.md) for the full I2V request format and supported models.

## See Also

- [Text-to-Video with SGLang](../text-to-video/sglang.md)
- [Image-to-Video with vLLM-Omni](vllm-omni.md)
- [SGLang Examples](../../../backends/sglang/sglang-examples.md)
