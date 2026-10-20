---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Image Understanding
subtitle: Serve vision language models that answer questions about images
sidebar-title: Overview
---

Dynamo serves vision language models (VLMs) through the OpenAI-compatible
`/v1/chat/completions` endpoint. A request carries one or more `image_url`
content parts next to its text, and the worker runs the model's vision encoder
before prefill.

This page gets one image request working on each backend. To generate images
instead, see [Text to Image](../diffusion/text-to-image.md). To speed up
image workloads, see [VLM Performance Optimization](vlm-performance-optimization.md).

## Support Matrix

| Input | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| `https://` image URL | Yes | Yes | Yes |
| `data:` (base64) image | Yes | No | Yes |
| Pre-computed embeddings (`.safetensors`) | No | No | Yes |

Plain `http://` URLs and hosts that resolve to private or loopback addresses are
refused by default. See [Media URL Policy](#media-url-policy).

## Set up the backend

Every multimodal worker must be started with `--enable-multimodal`. Without it,
media requests are rejected. The launchers below set the flag for you.

Run the commands from the root of a Dynamo repository checkout. Each launcher
starts a frontend on port 8000 and one aggregated worker on GPU 0.

<Tabs>
  <Tab title="vLLM" language="vllm">
    Requires a working [vLLM backend setup](../../developer-guide/knowledge-base/modular-components/backends/vllm/overview.md).

    ```bash
    bash examples/backends/vllm/launch/agg_multimodal.sh --model Qwen/Qwen3-VL-2B-Instruct
    ```

    The launcher defaults to `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` when `--model`
    is omitted. For XPU devices, use `launch/xpu/agg_multimodal_xpu.sh`. See
    [vLLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md)
    for disaggregated layouts, LoRA adapters, and supported models.
  </Tab>
  <Tab title="SGLang" language="sglang">
    Requires a working [SGLang backend setup](../../developer-guide/knowledge-base/modular-components/backends/sglang/overview.md).

    ```bash
    bash examples/backends/sglang/launch/agg_vision.sh --model-path Qwen/Qwen3-VL-2B-Instruct
    ```

    See [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md)
    for disaggregated layouts and supported models.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    Requires a working [TensorRT-LLM backend setup](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/overview.md).

    ```bash
    bash examples/backends/trtllm/launch/agg_multimodal.sh
    ```

    The launcher serves `Qwen/Qwen2-VL-7B-Instruct` with the engine
    configuration in `examples/backends/trtllm/engine_configs/qwen2-vl-7b-instruct/agg.yaml`.
    Set `"model": "Qwen/Qwen2-VL-7B-Instruct"` in the request below. See
    [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md)
    for Llama 4, disaggregated layouts, and pre-computed embeddings.
  </Tab>
</Tabs>

## Send an Image Request

In another terminal, send a chat request with an `image_url` content part:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}}
      ]
    }],
    "max_tokens": 64
  }'
```

To send a local file, base64-encode it as a `data:` URL on vLLM or
TensorRT-LLM: `"url": "data:image/jpeg;base64,<encoded bytes>"`.

## Media URL Policy

Workers fetch media URLs under a default-deny policy:

- `https://` and `data:` sources are allowed.
- Plain `http://` URLs and hostnames that resolve to private or loopback
  addresses are refused. Set `DYN_MM_ALLOW_INTERNAL=1` on the worker that loads
  the media to fetch from the cluster's internal network.
- A worker that reaches the internet through an HTTP proxy must set
  `DYN_MM_TRUST_EGRESS_PROXY=1`. Without it, a fetch that would go through the
  proxy is refused. Direct fetches, including hosts in `NO_PROXY`, are unaffected.
- Each remote download is capped at `DYN_MM_MAX_FILE_SIZE_MB` (default `64`).
  TensorRT-LLM uses its own `--max-file-size-mb` option instead.

> [!WARNING]
> `DYN_MM_ALLOW_INTERNAL=1` lets request authors make the worker fetch internal
> URLs. Enable it only when requests come from trusted clients.

## Next Steps

- [Custom Vision Encoders](custom-vision-encoders.md): run your own vision tower in a vLLM worker.

## Performance Optimizations

Once requests work, you can cut latency and raise throughput for image
workloads. [VLM Performance Optimization](vlm-performance-optimization.md) shows
where a request spends its time and when each feature helps:

- [Multimodal KV routing](vlm-performance-optimization/multimodal-kv-routing.md): send repeated images to the worker that already holds their KV cache.
- [Frontend media decoding](vlm-performance-optimization/frontend-media-decoding.md): decode images in the Rust frontend.
- [Shared image cache](vlm-performance-optimization/shared-image-cache.md): reuse downloaded image bytes across workers.
- [Embedding cache](vlm-performance-optimization/embedding-cache.md): skip the vision encoder for images it has seen.
- [Encoder disaggregation](vlm-performance-optimization/encoder-disaggregation.md): scale vision encoders separately from LLM workers.
