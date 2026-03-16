---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Route multimodal requests to workers with the best KV cache overlap
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image content when computing cache overlap scores. A dedicated MM router worker sits between the frontend and backend workers. It downloads images, computes a hash of each image (`mm_hash`), and includes this hash in per-block routing metadata. The KV router then selects the backend worker with the highest cache overlap, including overlap on image embedding blocks.

Repeated requests containing the same image are routed to the worker that already has the corresponding KV cache blocks, maximizing prefix cache reuse.
> Note: KV cache is separate from embedding cache (also called encoder cache), which reuses vision encoder outputs (image→embeddings) to avoid re-running the encoder. For encoder-side reuse see [Embedding Cache](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/embedding-cache.md).
## When to Use

Use multimodal KV routing when:

- You have multiple backend workers serving multimodal requests
- Your workload includes repeated images across requests (e.g., the same product photo, shared reference images)
- You want to maximize KV cache hit rates for multimodal content

Without MM-aware routing, the standard router treats image token blocks as opaque and cannot match which worker has cached a particular image's KV blocks.

## Support Matrix

| Backend | Supported | Notes |
|---------|-----------|-------|
| **vLLM** | ✅* | Requires vLLM with KV events `extra_keys` support ([PR #33304](https://github.com/vllm-project/vllm/pull/33304)) |
| **TRT-LLM** | ✅ | Requires `--publish-events-and-metrics` on TRT-LLM workers |
| **SGLang** | ❌ | Not supported yet |

*Requires an upcoming version of vLLM that has not yet been released. Support will be available once the new vLLM release is published.

## How It Works

```text
Frontend (round-robin) → MM Router Worker → Backend Workers
                              │
                              ├─ Download image
                              ├─ Compute mm_hash
                              ├─ Build per-block MM metadata
                              └─ KvRouter selects best worker
```

1. The frontend routes to the MM router worker via round-robin
2. The MM router downloads each image and computes an `mm_hash`
3. Per-block routing metadata (`block_mm_infos`) is built, tagging blocks that contain image tokens
4. The KV router evaluates overlap across all backend workers, accounting for image-bearing blocks
5. The request is forwarded to the worker with the highest overlap

On repeated requests with the same image, the selected worker shows higher cached block counts, reducing prefill latency.

## Launching

### vLLM

```bash
cd $DYNAMO_HOME/examples/backends/vllm/mm_router_worker
MODEL=Qwen/Qwen3-VL-2B-Instruct ./launch.sh
```

### TRT-LLM

```bash
cd $DYNAMO_HOME/examples/backends/trtllm/mm_router_worker
./launch.sh
```

See the [vLLM MM Router README](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/mm_router_worker/README.md) and [TRT-LLM MM Router README](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/mm_router_worker/README.md) for full setup instructions and configuration options.

## Known Limitations

- Currently supports Qwen-family multimodal processors (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) for per-image visual token counting
- Images are downloaded twice: once in the MM router (for hash computation) and once in the backend worker (for processing)
