---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Diffusion
subtitle: Deploy diffusion models for image, video, audio, and text generation
sidebar-title: Overview
---

## Overview

Dynamo serves diffusion models for text-to-image, text-to-video, image-to-video, text-to-audio, and text-to-text generation. These workloads use the same frontend, routing, scaling, and observability infrastructure as LLM inference.

The built-in backends expose OpenAI-compatible endpoints for images (`/v1/images/generations`), videos (`/v1/videos`), audio (`/v1/audio/speech`), and text generation.

<Info>
Built-in diffusion backends currently use Dynamo CLI launch scripts and `python -m dynamo.*` entrypoints. Dynamo does not yet ship prebuilt Kubernetes recipes for these backends. FastVideo is the exception and includes a Kubernetes deployment path.
</Info>

<Steps toc={true} tocDepth={2}>
  <Step title="Choose a Workflow" id="choose-a-workflow">
    Start with the type of content you want to generate.

    <CardGroup cols={2}>
      <Card title="Text-to-Image">
        Generate images from text prompts.

        **Backends:** [vLLM-Omni](text-to-image/vllm-omni.md), [SGLang](text-to-image/sglang.md), [TensorRT-LLM](text-to-image/trtllm.md)
      </Card>
      <Card title="Text-to-Video">
        Generate video clips from text prompts.

        **Backends:** [vLLM-Omni](text-to-video/vllm-omni.md), [SGLang](text-to-video/sglang.md), [TensorRT-LLM](text-to-video/trtllm.md), [FastVideo](text-to-video/fastvideo.md)
      </Card>
      <Card title="Image-to-Video">
        Animate a source image from a text prompt.

        **Backends:** [vLLM-Omni](image-to-video/vllm-omni.md), [SGLang](image-to-video/sglang.md)
      </Card>
      <Card title="Text-to-Audio">
        Generate speech from text.

        **Backend:** [vLLM-Omni](text-to-audio/vllm-omni.md)
      </Card>
      <Card title="Text-to-Text">
        Generate text through iterative diffusion rather than autoregressive decoding.

        **Backend:** [SGLang](text-to-text/sglang.md)
      </Card>
    </CardGroup>
  </Step>

  <Step title="Choose a Backend" id="choose-a-backend">
    Compare backend coverage and deployment constraints.

    <CardGroup cols={2}>
      <Card title="vLLM-Omni">
        <Badge intent="info" minimal>Broadest modality coverage</Badge>

        **Best for:** Mixed media workloads, text-to-audio, or disaggregated multi-stage serving.

        **Supports:** Text-to-image, text-to-video, image-to-video, and text-to-audio.

        **Main limitation:** Each worker serves one output modality at a time. Audio streaming and voice cloning are not yet supported.
      </Card>
      <Card title="SGLang">
        <Badge intent="info" minimal>LLM diffusion</Badge>

        **Best for:** Diffusion alongside an existing SGLang deployment or text-to-text diffusion models.

        **Supports:** Text-to-image, text-to-video, image-to-video, and text-to-text.

        **Main limitation:** Text-to-audio is not supported.
      </Card>
      <Card title="TensorRT-LLM">
        <Badge intent="warning" minimal>Experimental</Badge>

        **Best for:** NVIDIA-optimized image and video generation when experimental status is acceptable.

        **Supports:** Text-to-image and text-to-video.

        **Main limitation:** Not recommended for production. Video output requires an NVENC-capable NVIDIA GPU.
      </Card>
      <Card title="FastVideo">
        <Badge intent="success" minimal>Kubernetes</Badge>

        **Best for:** Fast, production-oriented text-to-video generation on Kubernetes.

        **Supports:** Text-to-video with a distilled LTX-2 model and five-step inference.

        **Main limitation:** Uses a purpose-built runtime image that can take 20–40 minutes or longer to build initially.
      </Card>
    </CardGroup>
  </Step>

  <Step title="Install the Backend" id="install-the-backend">
    Choose the tab for the backend you plan to use.

    <Tabs>
      <Tab title="vLLM-Omni">
        [vLLM-Omni](https://github.com/vllm-project/vllm-omni) provides image, video, and audio generation through `python -m dynamo.vllm.omni`.

        **Prerequisites**

        - A working [vLLM backend setup](../../backends/vllm/README.md).
        - An `amd64` host. Dynamo container builds do not install vLLM-Omni on `arm64`.

        Dynamo container images include vLLM-Omni. A PyPI installation of `ai-dynamo[vllm]` does not include it automatically. For a source installation, pin the vLLM-Omni release that matches your vLLM version:

        ```bash
        pip install git+https://github.com/vllm-project/vllm-omni.git@<version>
        ```

        <AccordionGroup>
          <Accordion title="Local launchers, media storage, and stage configuration">
            The `agg_omni_*.sh` and `disagg_omni_glm_image.sh` scripts launch the frontend and workers directly on one host. They are intended for local development and testing and do not create a `DynamoGraphDeployment`.

            Generated media uses [fsspec](https://filesystem-spec.readthedocs.io/) storage. The default is `file:///tmp/dynamo_media`. Set `--media-output-fs-url` for S3, GCS, or Azure Blob storage, and optionally set `--media-output-http-url` to rewrite response URLs.

            vLLM-Omni includes built-in YAML stage configurations for supported models. Set `--stage-configs-path` only to override them.
          </Accordion>
        </AccordionGroup>

        See the [vLLM-Omni Configuration reference](../../backends/vllm/vllm-omni-config-reference.mdx) for the complete flag surface.
      </Tab>

      <Tab title="SGLang">
        SGLang provides LLM diffusion, image diffusion, and video generation through its standard Dynamo backend.

        **Prerequisites**

        - A working [SGLang backend setup](../../backends/sglang/README.md).

        No separate installation is required. Select the worker mode for your workload:

        | Workload | Worker flag | Endpoint |
        |----------|-------------|----------|
        | Text-to-text | `--dllm-algorithm <algo>` | `/v1/chat/completions`, `/v1/completions` |
        | Text-to-image | `--image-diffusion-worker` | `/v1/images/generations` |
        | Text-to-video or image-to-video | `--video-generation-worker` | `/v1/videos` |

        <AccordionGroup>
          <Accordion title="Troubleshoot a CuDNN version mismatch">
            If startup reports `cuDNN frontend 1.8.1 requires cuDNN lib >= 9.5.0`, set:

            ```bash
            export SGLANG_DISABLE_CUDNN_CHECK=1
            ```

            This can occur when PyTorch includes an older CuDNN version than SGLang requires for Conv3d operations.
          </Accordion>
        </AccordionGroup>

        See the [SGLang Configuration reference](../../backends/sglang/sglang-config-reference.mdx).
      </Tab>

      <Tab title="TensorRT-LLM">
        TensorRT-LLM provides experimental image and video generation through `tensorrt_llm._torch.visual_gen`.

        **Prerequisites**

        - TensorRT-LLM installed with the `visual_gen` module.
        - A Dynamo runtime with image and video model types.
        - A GPU with enough VRAM for the selected model.

        Text-to-image requires no additional setup beyond TensorRT-LLM and Dynamo.

        <AccordionGroup>
          <Accordion title="Configure MP4 video encoding">
            Text-to-video output requires `imageio`, ffmpeg, and an NVENC-capable NVIDIA GPU. Dynamo's TensorRT-LLM runtime image includes an ffmpeg build with `h264_nvenc`.

            Outside the container, install the Python wrapper without its bundled binary and point it to your ffmpeg:

            ```bash
            pip install --no-binary imageio-ffmpeg "imageio[ffmpeg]"
            export IMAGEIO_FFMPEG_EXE=/path/to/your/ffmpeg
            ```
          </Accordion>
        </AccordionGroup>

        See the [TensorRT-LLM Configuration reference](../../backends/trtllm/trtllm-config-reference.mdx).
      </Tab>

      <Tab title="FastVideo">
        [FastVideo](https://github.com/hao-ai-lab/FastVideo) is a custom text-to-video worker that serves `/v1/videos` with a distilled LTX-2 model.

        **Prerequisites**

        - A GPU-enabled host or Kubernetes cluster.
        - An NVIDIA GPU runtime.
        - A container registry for the custom runtime image.

        <AccordionGroup>
          <Accordion title="Build and deploy FastVideo">
            Build the purpose-built runtime from [`examples/diffusers/Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile). The first build can take 20–40 minutes or longer because it installs FastVideo and compiles its dependencies.

            Kubernetes is the recommended deployment path. Follow the [FastVideo guide](text-to-video/fastvideo.md) for image build, deployment, and configuration instructions.
          </Accordion>
        </AccordionGroup>
      </Tab>
    </Tabs>
  </Step>
</Steps>

## Support Matrix

| Workflow | vLLM-Omni | SGLang | TensorRT-LLM | FastVideo |
|----------|-----------|--------|--------------|-----------|
| Text-to-Image | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="warning" minimal>Experimental</Badge> | — |
| Text-to-Video | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="warning" minimal>Experimental</Badge> | <Badge intent="success" minimal>Yes</Badge> |
| Image-to-Video | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | — | — |
| Text-to-Audio | <Badge intent="success" minimal>Yes</Badge> | — | — | — |
| Text-to-Text | — | <Badge intent="success" minimal>Yes</Badge> | — | — |
