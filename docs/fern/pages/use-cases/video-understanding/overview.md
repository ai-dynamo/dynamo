---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Understanding
subtitle: Serve vision language models that answer questions about video clips
sidebar-title: Overview
---

Dynamo serves video-capable vision language models through the OpenAI-compatible
`/v1/chat/completions` endpoint. A request carries a `video_url` content part
next to its text. Dynamo decodes the clip, samples frames, and the worker runs
the model's vision encoder on them.

This page gets one video request working on each backend. To generate video
instead, see [Text to Video](../diffusion/text-to-video.md). To route repeated clips to the
worker that already holds their KV cache, see
[Exact Video KV Routing](exact-video-kv-routing.md).

## Support Matrix

Which codecs work depends on where the video is decoded:

| Backend | H.264 / H.265 | VP8 / VP9 | Notes |
| --- | --- | --- | --- |
| vLLM | Yes (NVDEC on the GPU) | Yes, with `--frontend-decoding` | |
| SGLang | Disaggregated Qwen2-family only (NVDEC) | Yes, aggregated with `--frontend-decoding` on CUDA | Aggregated VP8/VP9 is validated for Qwen3-VL and Qwen3.5 |
| TensorRT-LLM | Yes (NVDEC on the GPU) | Through the TensorRT-LLM loader, after you [install a software decoder](../image-understanding/additional-media-decoders.md) | |

- **NVDEC** is the GPU's hardware video decoder. The container must be granted
  the `video` driver capability, or hardware decode is silently unavailable.
- **Frontend decoding** (`--frontend-decoding` on the worker) decodes VP8 and VP9
  on the CPU in Dynamo's Rust frontend. It does not decode H.264, H.265, or AV1.
- The runtime images ship no software H.264, H.265, or AV1 decoder.

See [Video Decoding and GPU Requirements](video-decoding.md) for GPU support,
container setup, and installing additional decoders.

## Set up the backend

The vLLM and TensorRT-LLM launchers below start the worker with
`--enable-multimodal`. The SGLang aggregated launcher (`agg_vision.sh`) does not
pass it. Run the commands from the root of a Dynamo repository
checkout. Each launcher starts a frontend on port 8000 and one aggregated worker.

<Tabs>
  <Tab title="vLLM" language="vllm">
    Requires a working [vLLM backend setup](../../developer-guide/knowledge-base/modular-components/backends/vllm/overview.md).

    ```bash
    bash examples/backends/vllm/launch/agg_multimodal.sh --model Qwen/Qwen3-VL-2B-Instruct
    ```

    To accept VP8 and VP9 input, append `--frontend-decoding`; the launcher
    passes extra arguments through to the worker:

    ```bash
    bash examples/backends/vllm/launch/agg_multimodal.sh --model Qwen/Qwen3-VL-2B-Instruct --frontend-decoding
    ```
  </Tab>
  <Tab title="SGLang" language="sglang">
    Requires a working [SGLang backend setup](../../developer-guide/knowledge-base/modular-components/backends/sglang/overview.md).

    ```bash
    bash examples/backends/sglang/launch/agg_vision.sh --model-path Qwen/Qwen3-VL-2B-Instruct --frontend-decoding
    ```

    Aggregated video on SGLang requires `--frontend-decoding` and a CUDA
    device. SGLang XPU images omit the frontend decoder. For H.264 or H.265 on
    Qwen2-family models, use the disaggregated encode-worker flow in
    [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md).
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    Requires a working [TensorRT-LLM backend setup](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/overview.md).

    ```bash
    bash examples/backends/trtllm/launch/agg_multimodal.sh
    ```

    The launcher serves `Qwen/Qwen2-VL-7B-Instruct`. Set
    `"model": "Qwen/Qwen2-VL-7B-Instruct"` in the request below. See
    [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md)
    for other models.

    TensorRT-LLM has no frontend video decoding, and the runtime image ships no
    software VP8/VP9 decoder. The example clip below is VP8/VP9, so on
    TensorRT-LLM it needs a software decoder installed as described in
    [Install Additional Decoders](video-decoding.md#install-additional-decoders),
    or an H.264 or H.265 clip of your own.
  </Tab>
</Tabs>

## Send a Video Request

In another terminal, send a chat request with a `video_url` content part. This
example clip is VP8/VP9 WebM, so it needs `--frontend-decoding` on vLLM and
SGLang, or a separately installed software decoder on TensorRT-LLM:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video."},
        {"type": "video_url", "video_url": {"url": "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm"}}
      ]
    }],
    "max_tokens": 64
  }'
```

`DYN_MM_VIDEO_NUM_FRAMES` (default `32`) sets the maximum number of frames
sampled from each clip. Keep it the same on the frontend and every worker.

Video URLs are fetched under the same default-deny policy as images: `https://`
and `data:` sources pass, and internal or plain `http://` hosts need
`DYN_MM_ALLOW_INTERNAL=1`. See
[Image Understanding › Media URL Policy](../image-understanding/overview.md#media-url-policy).

## Next Steps

- [Exact Video KV Routing](exact-video-kv-routing.md): reuse KV cache for repeated clips across workers.
- [Video Decoding and GPU Requirements](video-decoding.md): NVDEC, container capabilities, and extra decoders.

## Performance Optimizations

Video requests share most of their performance features with image requests.
[VLM Performance Optimization](../image-understanding/vlm-performance-optimization.md)
shows where a request spends its time and when each feature helps. The ones that
apply to video:

- [Frontend media decoding](../image-understanding/vlm-performance-optimization/frontend-media-decoding.md): decode VP8/VP9 video in the Rust frontend.
- [Embedding cache](../image-understanding/vlm-performance-optimization/embedding-cache.md): skip the vision encoder for clips it has seen.
- [Multimodal KV routing](../image-understanding/vlm-performance-optimization/multimodal-kv-routing.md) and [Exact Video KV Routing](exact-video-kv-routing.md): send repeated clips to the worker that already holds their KV cache.
