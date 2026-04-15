---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Route multimodal requests to workers with the best KV cache overlap
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image content when computing cache overlap scores. The frontend runs vLLM's HF image processor to compute a hash of each image (`mm_hash`) and includes this hash in per-block routing metadata. The KV router then selects the backend worker with the highest cache overlap, including overlap on image embedding blocks.

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
| **vLLM** | ✅ | Uses frontend vLLM processor with KV router (`--dyn-chat-processor vllm --router-mode kv`) |
| **TRT-LLM** | ✅ | Uses dedicated MM Router Worker. Requires `--publish-events-and-metrics` on TRT-LLM workers |
| **SGLang** | ❌ | Not supported yet |

## How It Works

### vLLM

```text
Frontend (vLLM processor + KV router) → Backend Workers
        │
        ├─ Download image (via DynamoMediaConnector, LRU cached)
        ├─ Run vLLM's process_inputs() (HF processor, model-agnostic)
        ├─ Extract mm_hash from mm_features
        ├─ Build per-block MM metadata (block_mm_infos)
        ├─ KV router selects best worker
        └─ Transfer pre-processed mm_kwargs via NIXL
              → Backend skips HF processor
```

1. The frontend's vLLM processor downloads images and runs `process_inputs()` — this invokes the HF image processor and produces expanded token IDs, mm_hashes, and processed pixel values
2. Per-block routing metadata (`block_mm_infos`) is built from the mm_features, tagging blocks that contain image tokens with their mm_hash
3. The KV router evaluates overlap across all backend workers, accounting for image-bearing blocks
4. Pre-processed `mm_kwargs` (pixel values, image grid info) are transferred to the selected worker via NIXL, so the backend skips the HF processor entirely
5. The backend injects the received kwargs into its processor cache for accurate MM cache hit rate metrics

On repeated requests with the same image, the selected worker shows higher cached block counts, reducing prefill latency.

**Key advantages:**

- **Model-agnostic**: Uses vLLM's own `process_inputs()` — supports all multimodal models that vLLM supports, with no model-specific token expansion code
- **No double processing**: Images are downloaded and processed once on the frontend; the backend receives pre-processed tensors via NIXL
- **In-process KV router**: No cross-process RPC overhead for routing decisions

### TRT-LLM

```text
Frontend (round-robin) → MM Router Worker → Backend Workers
                              │
                              ├─ Download image
                              ├─ Compute mm_hash
                              ├─ Build per-block MM metadata
                              └─ KvRouter selects best worker
```

For TRT-LLM, a dedicated MM Router Worker sits between the frontend and backend workers. See the [TRT-LLM MM Router README](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/mm_router_worker/README.md) for setup instructions.

## Launching

### vLLM

```bash
cd $DYNAMO_HOME
bash examples/backends/vllm/launch/agg_multimodal_router.sh
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Model to serve |
| `NUM_WORKERS` | `2` | Number of backend workers |
| `BLOCK_SIZE` | `16` | KV cache block size (must match backend) |
| `GPU_MEMORY_UTILIZATION` | `0.40` | Per-worker GPU memory fraction |

### TRT-LLM

```bash
cd $DYNAMO_HOME/examples/backends/trtllm/mm_router_worker
./launch.sh
```

See the [TRT-LLM MM Router README](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/mm_router_worker/README.md) for full setup instructions and configuration options.

## Known Limitations

- Accurate MM cache hit rate metrics require vLLM's `InputProcessor.inject_into_mm_cache()` API (upstream PR pending).
