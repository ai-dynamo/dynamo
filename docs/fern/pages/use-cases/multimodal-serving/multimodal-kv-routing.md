---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Include image and video identity in the router's combined cache-and-load cost
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image or video content when calculating cache overlap. The frontend assigns each supported media object a stable hash and includes its identity in the routing view of the prompt.

When media appears again, its identity contributes to each worker's KV overlap. The router balances that cache credit against projected prefill and decode load, so the worker with the largest overlap does not necessarily win when it is busy. Cache-aware placement increases prefix reuse without abandoning load balancing.

> [!IMPORTANT]
> The KV cache stores attention key/value state so a worker can skip repeated prefill work. The embedding cache stores vision encoder outputs so the encoder can skip repeated image processing. You can use both features together. See [Embedding Cache](embedding-cache.md).

## When to Use

Use multimodal KV routing when:

- Multiple backend workers serve multimodal requests.
- Images or videos repeat across requests, such as product photos, shared reference images, or several questions about the same video.
- You want to maximize KV cache reuse for multimodal content.

Single-worker deployments do not need routing, and workloads with entirely unique media receive little media-specific cache benefit. Exact video routing decodes and samples the video in the frontend before worker selection, so use it when the expected KV reuse offsets that frontend work.

## How It Works

The routing flow in general has three steps:

1. The frontend computes a stable identity for each supported media object.
2. The frontend represents the media in a routing-only token view that matches the backend's cache identity and token layout.
3. The KV router includes that overlap in its combined cache-and-load score, selects the lowest-cost eligible worker, and forwards the same media identity.

### Image Identity

By default, an HTTP image's identity hashes the exact URL bytes, including its
query string. Reusing the identical URL produces the same identity, but two URLs
for the same image bytes do not. Enable `--frontend-decoding` when you need
content-stable identity across URLs; the frontend then hashes decoded image
content. Data URLs follow the same rule: the default path hashes the full data
URI string, while frontend decoding hashes the decoded bytes.

### Video Identity and Token Layout

Exact video routing requires frontend decoding. The frontend samples the video and computes an XXH3-64 identity over the sampled RGB frames, decoded shape and data type, and video metadata such as the source frame rate and sampled timestamps. Two URLs that decode to the same sampled content and metadata receive the same identity.

Video prompt expansion is model-specific. The frontend uses a model adapter to reproduce the worker's resize, patch, temporal grouping, delimiter, timestamp, and pruning behavior in a routing-only token sequence. At worker startup, vLLM reports the installed processor behavior that affects this sequence. Dynamo disables exact video routing if the reported contract does not match the frontend adapter.

The selected worker receives the same per-video hash, and its KV events use that identity. This keeps the frontend's routing blocks aligned with the blocks published by vLLM without running the vision model in the frontend.

<Tabs>
  <Tab title="vLLM" language="vllm">
    vLLM provides two routing paths. Use the default Rust frontend for supported model families when you want minimal frontend processing. Use the Python chat processor when you need vLLM's broader model support or want the frontend to preprocess media and transfer the processed inputs to workers.

    **Default Rust frontend**

    For images, the frontend calculates only the identity and routing token layout unless frontend decoding is enabled. For videos, exact routing requires `--frontend-decoding` so the frontend can sample the video and build its model-visible token layout. The selected worker still runs the model's multimodal processor. This path depends on the model being registered with Dynamo's Rust processor registry.

    **Alternative: Python chat processor**

    With `--dyn-chat-processor vllm`, the frontend runs vLLM's full multimodal processor. It derives routing positions from vLLM's processed image or video features without requiring a Dynamo Rust processor specification and can transfer processed inputs through shared memory or NIXL. This shifts preprocessing and transfer work to the frontend.

  </Tab>
  <Tab title="SGLang" language="sglang">
    The frontend hashes each image and converts that hash into the same `pad_value` token that SGLang RadixAttention uses for its prefix-cache key. The matching token view lets the router measure image overlap before selecting a worker.

    Dynamo's SGLang image includes the required hash-forwarding support. Exact video routing is not available on this path; video requests use text-prefix routing.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    The frontend hashes each image, represents that identity in its routing token view, and forwards the hash as `multi_modal_uuids`. TensorRT-LLM workers publish matching KV events so the router can identify cached image blocks.

    Exact video routing is not available on this path; video requests use text-prefix routing.

  </Tab>
</Tabs>

## Launch

<Tabs>
  <Tab title="vLLM" language="vllm">
    Use the Python chat processor when you need vLLM's model-native multimodal processor or want to transfer processed multimodal inputs from the frontend. Otherwise, use the default Rust frontend.

    **Default Rust frontend**

    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/vllm/launch/agg_multimodal_router.sh
    ```

    To enable exact video routing on the default path, add frontend decoding:

    ```bash
    cd $DYNAMO_HOME
    MODEL=Qwen/Qwen3-VL-2B-Instruct \
      NUM_WORKERS=2 \
      VLLM_EXTRA_ARGS="--frontend-decoding" \
      bash examples/backends/vllm/launch/agg_multimodal_router.sh
    ```

    `DYN_MM_VIDEO_NUM_FRAMES` controls the maximum number of sampled frames and defaults to `32`. Keep this setting consistent across workers. The frontend passes the decoded frames and sampling metadata to the selected worker, so the worker processes the same view used for routing.

    **Alternative: Python chat processor**

    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/vllm/launch/agg_multimodal_router_chat_processor.sh
    ```

    See [vLLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) for model support, hashing behavior, transfer modes, and configuration.
  </Tab>
  <Tab title="SGLang" language="sglang">
    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/sglang/launch/agg_multimodal_router.sh
    ```

    The launcher configures KV events and matching frontend and worker block sizes.

    See [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#multimodal-kv-routing) for prerequisites, configuration, fallback behavior, and verification.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/trtllm/launch/agg_multimodal_router.sh
    ```

    The launcher enables multimodal serving, KV event publishing, block reuse, and KV-aware routing.

    See [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md#multimodal-kv-routing) for worker requirements, supported models, and limitations.
  </Tab>
</Tabs>

## Send a Video Request

The standard OpenAI chat payload uses a `video_url` content part:

```bash
VIDEO_URL="https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm"

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this video."},
      {"type": "video_url", "video_url": {"url": "${VIDEO_URL}"}}
    ]
  }],
  "max_tokens": 32
}
JSON
```

Send the request twice to populate and then reuse the worker's KV cache. Exact routing through the default Rust frontend supports VP8 or VP9 video in MP4, WebM, or MKV containers. The in-tree frontend decoder does not support H.264 or H.265. Re-encode those inputs to VP9, or disable frontend decoding and let the backend decode them through NVDEC or an installed software decoder; with backend-only decoding, video identity does not contribute to exact routing. See [Additional Media Decoders](additional-media-decoders.md) and [Video Decode GPU Requirements](video-decode-gpu-requirements.md).

## Verify Video Routing

The vLLM multimodal router launcher enables routing debug logs by default. For a custom launch, set `DYN_LOG=info,mm_routing=debug,dynamo_kv_router::scheduling=debug,dynamo_llm::kv_router=debug`, then check for:

- `exact video-aware KV routing enabled` during worker registration.
- `video routing metadata resolved` after the frontend decodes an eligible request.
- `[ROUTING]` records with `worker_id`, `overlap_blocks`, and `total_blocks` for each request.

With idle workers, a warm repeat should select the same worker and report more overlap than the first request. A different video should not receive the first video's full overlap. The router can still choose a less-cached worker when its combined cache-and-load cost is lower.

If startup or request validation cannot prove that the frontend and worker produce the same token layout, Dynamo logs the reason and uses text-prefix routing. The request continues to run, but video identity does not contribute to worker selection.

## Exact Video Routing Limits

The default Rust frontend supports exact video routing for:

- Qwen3-VL and Qwen3.5 dense and mixture-of-experts models. Video pruning is not supported on this path.
- Nemotron 3 Nano Omni. The adapter reproduces temporal grouping, frame separators, dynamic resolution, and optional EVS pruning.

Exact video routing falls back to text-prefix routing for unsupported models, opaque client-provided UUIDs, non-empty `mm_processor_kwargs`, audio in the same request, or adjacent video objects. Nemotron uses the same `<image>` placeholder for image and video features, so its exact path currently supports one video-only media object per request; mixed image-and-video or multiple-video Nemotron requests fall back.

## Support Matrix

| Backend | Routing Path | Status | Notes |
|---------|--------------|--------|-------|
| [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Exact image routing supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA 1.5, LLaVA-NeXT, Llama 4, Kimi K2.5/K2.6, Qwen3.5, Qwen3.6, and Nemotron 3 Nano Omni. Exact video routing requires frontend decoding and supports Qwen3-VL, Qwen3.5, and Nemotron 3 Nano Omni. Other media layouts use text-prefix routing. |
| [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) | Python chat processor | <Badge intent="success" minimal>Yes</Badge> | Uses vLLM's multimodal processor to derive image and video feature hashes and positions for models supported by vLLM. |
| [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Images</Badge> | Hash forwarding is upstream in SGLang 0.5.13+; Dynamo pins 0.5.19. Video requests use text-prefix routing. |
| [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Images</Badge> | Exact image routing supports the Qwen2-VL family and Kimi K2.5/K2.6. Video requests and other models use text-prefix routing. |

Nemotron 3 Nano Omni exact image and video routing currently requires the vLLM backend. Other backends fall back to text-prefix routing for this model.
