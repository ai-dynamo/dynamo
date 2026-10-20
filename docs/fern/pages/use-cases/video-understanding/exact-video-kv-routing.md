---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Exact Video KV Routing
subtitle: Route repeated video to the worker that already holds its KV cache
---

Exact video KV routing extends [Multimodal KV Routing](../image-understanding/vlm-performance-optimization/multimodal-kv-routing.md)
to video input. The frontend decodes and samples each video, computes a stable
identity for it, and reproduces the worker's video token layout, so the KV router
can credit cached video blocks when it picks a worker.

Use it when several workers serve video requests and the same clips repeat
across requests, such as several questions about one video. Exact video routing
decodes and samples the video in the frontend before worker selection, so use it
when the expected KV reuse offsets that frontend work.

![A request with a repeated video reaches the frontend. The frontend decodes and samples the video, hashes the decoded content, reproduces the worker's token layout, builds a routing-only prompt, and scores workers. Worker A, which holds the clip's KV blocks, is selected.](../../../assets/img/exact-video-kv-routing.svg)

## Video Identity and Token Layout

Exact video routing requires frontend decoding. The frontend samples the video and computes an XXH3-64 identity over the sampled RGB frames, decoded shape and data type, and video metadata such as the source frame rate and sampled timestamps. Two URLs that decode to the same sampled content and metadata receive the same identity.

Video prompt expansion is model-specific. The frontend uses a model adapter to reproduce the worker's resize, patch, temporal grouping, delimiter, timestamp, and pruning behavior in a routing-only token sequence. At worker startup, vLLM and eligible SGLang workers report the installed processor behavior that affects this sequence. Dynamo disables exact video routing if the reported contract does not match the frontend adapter.

The selected worker receives the same per-video hash, and its KV events use that identity. This keeps the frontend's routing blocks aligned with the blocks published by the backend without running the vision model in the frontend.

## Launch

Exact video routing requires [frontend decoding](../image-understanding/vlm-performance-optimization/frontend-media-decoding.md)
(`--frontend-decoding`) on the worker.

<Tabs>
  <Tab title="vLLM" language="vllm">
    ```bash
    cd $DYNAMO_HOME
    MODEL=Qwen/Qwen3-VL-2B-Instruct \
      NUM_WORKERS=2 \
      VLLM_EXTRA_ARGS="--frontend-decoding" \
      bash examples/backends/vllm/launch/agg_multimodal_router.sh
    ```

    `DYN_MM_VIDEO_NUM_FRAMES` sets the maximum number of sampled frames and defaults to `32`. Keep the value the same on every worker. The frontend passes the decoded frames and sampling metadata to the selected worker, so the worker processes the same view used for routing.
  </Tab>
  <Tab title="SGLang" language="sglang">
    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/sglang/launch/agg_multimodal_router.sh
    ```

    The launcher enables `--frontend-decoding`, configures KV events, and sets matching frontend and worker block sizes. Set `DYN_MM_VIDEO_NUM_FRAMES` before launching to change the maximum sampled frame count from its default of `32`; keep the same value for the frontend and every worker.

    Video routing on SGLang supports aggregated Qwen3-VL and Qwen3.5 workers on CUDA. The worker publishes its effective frame-sampling, resize, timestamp, and placeholder contract; exact video routing falls back to text-prefix routing when a worker lacks support, the discovery cohort contains legacy or mismatched contracts, or the installed processor behavior is incompatible.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    Exact video routing is not available on TensorRT-LLM. Video requests use text-prefix routing.
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

Send the request twice to populate and then reuse the worker's KV cache. Exact routing through the default Rust frontend supports VP8 or VP9 video in MP4, WebM, or MKV containers. SGLang provides this decoder on CUDA only. The in-tree frontend decoder does not support H.264 or H.265. Re-encode those inputs to VP9, or use a backend-supported NVDEC or software-decoder path; with backend-only decoding, video identity does not contribute to exact routing. See [Video Decoding and GPU Requirements](video-decoding.md).

## Verify Video Routing

The vLLM and SGLang multimodal router launchers enable routing debug logs by default. For a custom launch, set `DYN_LOG=info,mm_routing=debug,dynamo_kv_router::scheduling=debug,dynamo_llm::kv_router=debug`, then check for:

- `exact video-aware KV routing enabled` during worker registration.
- `video routing metadata resolved` after the frontend decodes an eligible request.
- `[ROUTING]` records with `worker_id`, `overlap_blocks`, and `total_blocks` for each request.

With idle workers, a warm repeat should select the same worker and report more overlap than the first request. A different video should not receive the first video's full overlap. The router can still choose a less-cached worker when its combined cache-and-load cost is lower.

If startup or request validation cannot prove that the frontend and worker produce the same token layout, Dynamo logs the reason and uses text-prefix routing. The request continues to run, but video identity does not contribute to worker selection.

## Exact Video Routing Limits

The default Rust frontend supports exact video routing for:

- vLLM Qwen3-VL and Qwen3.5 dense and mixture-of-experts models. Video pruning is not supported on this path.
- vLLM Nemotron 3 Nano Omni. The adapter reproduces temporal grouping, frame separators, dynamic resolution, and optional EVS pruning.
- Aggregated SGLang Qwen3-VL and Qwen3.5 dense and mixture-of-experts models on CUDA with `--frontend-decoding`.

Exact video routing falls back to text-prefix routing for unsupported models, opaque client-provided UUIDs, non-empty `mm_processor_kwargs`, audio in the same request, or adjacent video objects. SGLang also falls back when workers publish missing, incompatible, or different processor contracts. Nemotron uses the same `<image>` placeholder for image and video features, so its exact path currently supports one video-only media object per request; mixed image-and-video or multiple-video Nemotron requests fall back.
