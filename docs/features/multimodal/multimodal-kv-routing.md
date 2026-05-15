---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Route multimodal requests to workers with the best KV cache overlap
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image content when computing cache overlap scores. An image hash (`mm_hash`) is computed per request — in the Rust frontend by default for vLLM backends, by vLLM's own processor when the chat-processor variant is enabled, or by a dedicated MM router worker for TRT-LLM backends — and included in per-block routing metadata. The KV router then selects the backend worker with the highest cache overlap, including overlap on image embedding blocks.

Repeated requests containing the same image are routed to the worker that already has the corresponding KV cache blocks, maximizing prefix cache reuse.

> Note: KV cache is separate from embedding cache (also called encoder cache), which reuses vision encoder outputs (image→embeddings) to avoid re-running the encoder. For encoder-side reuse see [Embedding Cache](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/embedding-cache.md).

## When to Use

Use multimodal KV routing when:

- You have multiple backend workers serving multimodal requests
- Your workload includes repeated images across requests (e.g., the same product photo, shared reference images)
- You want to maximize KV cache hit rates for multimodal content

Without MM-aware routing, the standard router treats image token blocks as opaque and cannot match which worker has cached a particular image's KV blocks.

## Support Matrix

| Backend | Path | Supported | Notes |
|---------|------|-----------|-------|
| **vLLM** | Rust frontend (default) | ✅ | Uses lightseek `llm-multimodal` for image-token counting + placeholder expansion. Supported models tracked below. |
| **vLLM** | Python chat-processor (`--dyn-chat-processor vllm --router-mode kv`) | ✅ | Uses vLLM's own multimodal processor — supports any VLM that vLLM supports. |
| **TRT-LLM** | — | ✅ | Uses dedicated MM Router Worker. Requires `--publish-events-and-metrics` on TRT-LLM workers. |
| **SGLang** | — | ❌ | Not supported yet. |

## Supported Model Families (Rust frontend path)

The Rust frontend's MM-aware routing path supports whatever VLM families the
lightseek `llm-multimodal` crate registers — see
[`ImageProcessorRegistry::with_defaults()`](https://docs.rs/llm-multimodal/1.5.0/llm_multimodal/vision/image_processor/struct.ImageProcessorRegistry.html#method.with_defaults)
for the up-to-date list. A model that crate doesn't recognize falls back to
text-prefix-only KV routing (request still completes; just no prefix-cache
benefit across images).

The Python chat-processor variant doesn't share this constraint — it
delegates to vLLM's own multimodal processor and works with any VLM vLLM
supports.

## How It Works

### vLLM (default — Rust frontend)

```text
Frontend (Rust + lightseek llm-multimodal + KV router) → Backend Workers
        │
        ├─ Hash image (xxh3_64 of the raw URL — full-URL identity; use --frontend-decoding for content-addressed hashing)
        ├─ Resolve image-token id via lightseek's per-model ModelProcessorSpec
        ├─ Read (W, H) from a Range: 0-65535 header fetch (or in-memory data: bytes)
        ├─ lightseek::count_tokens(W, H) → expanded image-token count N
        ├─ Expand placeholder × N in routing_token_ids (worker token_ids unchanged)
        ├─ Build per-block MM metadata (block_mm_infos)
        ├─ KV router selects best worker
        └─ Forward mm_hash to worker via extra_args["mm_hashes"] →
              vLLM's multi_modal_uuids (cache key match)
```

1. The Rust frontend computes an `mm_hash` per image: `xxh3_64` of the decoded bytes for `data:` URIs (and for `http(s)://` when `media_decoder` is enabled on the model), otherwise `xxh3_64` of the full URL string. Two callers will share an `mm_hash` only when they send byte-identical URLs.
2. The image-placeholder token id is resolved by delegating to lightseek's per-model `ModelProcessorSpec` (one spec per supported VLM family — Qwen3-VL, Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT, LLaVA-1.5, Phi-3-vision, Llama-4, Kimi-K2.5). Each spec reads the appropriate `config.json` field for its model family (`image_token_id`, `image_token_index`, or `media_placeholder_token_id`) and falls back to probing the tokenizer's vocab when only the placeholder string is registered. Models the registry doesn't recognise fall back to text-prefix-only routing.

> **Note:** Qwen3.5 / Qwen3.6 image token expansion is not yet supported in the Rust frontend for MM routing, so KV routing will only consider the text inputs + unexpanded image token placeholders. Support will come in a follow-up release.
3. Per-image `(W, H)` is read from a 64KB `Range`-bounded header fetch (or from in-memory bytes for `data:` URIs); the lightseek `llm-multimodal` crate computes the per-image expanded token count.
4. The single placeholder token is expanded to N copies in `routing_token_ids` (a router-only view); the worker still sees one placeholder per image in `token_ids`.
5. Per-block MM metadata (`block_mm_infos`) is built from the expanded view; the KV router evaluates overlap across workers including image-bearing blocks.
6. The frontend forwards each image's `mm_hash` (16-hex-char prefix, padded) via `extra_args["mm_hashes"]`; the backend handler injects them as vLLM's `multi_modal_uuids`, so vLLM's own KV-cache key matches the hash the router used.

### vLLM (alternative — Python chat-processor variant)

```text
Frontend (vLLM processor + KV router) → Backend Workers
        │
        ├─ Download image (via DynamoMediaConnector, LRU cached)
        ├─ Run vLLM's process_inputs() (HF processor, model-agnostic)
        ├─ Extract mm_hash from mm_features
        ├─ Build per-block MM metadata (block_mm_infos)
        ├─ KV router selects best worker
        └─ Transfer pre-processed mm_kwargs via SHM or NIXL
              → Backend skips HF processor
```

Use this variant (`--dyn-chat-processor=vllm`) when you want the frontend to run vLLM's HF image processor in-process and ship pre-processed `mm_kwargs` to the selected worker via shared memory or NIXL RDMA, so the backend skips the HF processor entirely. See the [Transfer Mode Details](#transfer-mode-details-vllm-only) section below for the `DYNAMO_MM_TRANSFER` flags.

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

### vLLM (default — Rust frontend)

```bash
cd $DYNAMO_HOME
bash examples/backends/vllm/launch/agg_multimodal_router.sh
```

The Rust frontend uses the [lightseek `llm-multimodal`][lightseek-crate] crate
([source][lightseek-src]) for per-image token-count and placeholder
expansion. `llm-multimodal` provides a pure-Rust `calculate_num_tokens(W, H,
PreProcessorConfig)` per VLM family (Qwen2/2.5/3-VL, LLaVA, Pixtral, …),
golden-tested against `transformers`, so the router can match vLLM's
expanded image-token count without invoking the HF image processor. The
frontend then forwards each `mm_hash` to the worker as `multi_modal_uuids`
so vLLM's KV events publish the same key the router computes.

[lightseek-crate]: https://crates.io/crates/llm-multimodal
[lightseek-src]: https://github.com/lightseekorg/smg

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Model to serve |
| `NUM_WORKERS` | `2` | Number of backend workers |
| `BLOCK_SIZE` | `16` | KV cache block size (must match backend) |
| `GPU_MEMORY_UTILIZATION` | `0.20` | Per-worker GPU memory fraction |
| `SINGLE_GPU` | `false` | Pack all workers onto GPU 0 (testing-only override; pass `--single-gpu` or set `SINGLE_GPU=true` for functional tests on a single-GPU box) |
| `KV_EVENTS_PORT_BASE` | `5557` | Worker `i` publishes ZMQ KV events on `BASE + i - 1` |
| `DYN_LOG` | `info,lightseek_mm=debug,...` | Frontend log filter |
| `VLLM_EXTRA_ARGS` | (unset) | Pass-through args to `python -m dynamo.vllm`. Set `--frontend-decoding` to enable content-addressed `mm_hash` (cross-URL KV-cache reuse). |

To opt into frontend image decoding (so the frontend downloads + decodes once and `mm_hash` becomes content-addressed instead of URL-addressed):

```bash
VLLM_EXTRA_ARGS="--frontend-decoding" \
    bash examples/backends/vllm/launch/agg_multimodal_router.sh
```

The worker then registers a `media_decoder` on its model card; the frontend's `MediaLoader` runs in-process and hashes decoded RGB bytes via xxh3. Two distinct (signed) URLs of the same image bytes collide on the same routing key.

### vLLM (alternative — Python chat-processor variant)

```bash
bash examples/backends/vllm/launch/agg_multimodal_router_chat_processor.sh
```

Uses `--dyn-chat-processor=vllm` so the frontend runs vLLM's HF processor
in-process. Adds the `DYNAMO_MM_TRANSFER` shm/NIXL pre-rendered `mm_kwargs`
delivery channel between frontend and worker.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Model to serve |
| `NUM_WORKERS` | `2` | Number of backend workers |
| `BLOCK_SIZE` | `16` | KV cache block size (must match backend) |
| `GPU_MEMORY_UTILIZATION` | `0.40` | Per-worker GPU memory fraction |
| `SINGLE_GPU` | `false` | Pack all workers onto GPU 0 (testing-only override; pass `--single-gpu` or set `SINGLE_GPU=true` for functional tests on a single-GPU box) |
| `DYNAMO_MM_TRANSFER` | `shm` | Transfer mode for pre-processed mm_kwargs: `shm` (shared memory, same-node), `nixl` (RDMA, cross-node) |
| `DYNAMO_DISABLE_NIXL_MM` | unset | Set to `1` to disable mm_kwargs transfer entirely (backend re-processes images from URLs) |

### TRT-LLM

```bash
cd $DYNAMO_HOME/examples/backends/trtllm/mm_router_worker
./launch.sh
```

See the [TRT-LLM MM Router README](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/mm_router_worker/README.md) for full setup instructions and configuration options.

## Transfer Mode Details (vLLM chat-processor variant only)

Applies to the `--dyn-chat-processor=vllm` launch (`agg_multimodal_router_chat_processor.sh`), **not** the default Rust frontend path. In the chat-processor variant the frontend runs the HF image processor in-process and ships the pre-processed `mm_kwargs` to the selected backend worker so the backend can skip re-processing; the `DYNAMO_MM_TRANSFER` environment variable controls how that payload is transferred.

The default Rust frontend path doesn't run the HF processor or pre-render `mm_kwargs` — it forwards only `mm_hashes`, and each worker re-processes the image itself. TRT-LLM backends similarly re-run their own preprocessing and don't honor `DYNAMO_MM_TRANSFER`.

- **`shm`** (default): POSIX shared memory via a `/dev/shm` segment. Intended for same-node deployments, where frontend and backend share the host filesystem. If the backend can't access the segment (e.g., running on a different node), it falls back to re-processing the image from the URL.
- **`nixl`**: NIXL RDMA transfer. Required for cross-node deployments where `/dev/shm` is not shared between frontend and backend. Works across nodes over InfiniBand or TCP (whichever UCX selects).
- **`DYNAMO_DISABLE_NIXL_MM=1`**: Disables pre-processed mm_kwargs transfer entirely. The backend downloads and processes images itself from the original URLs. Useful for debugging or when transfer overhead exceeds re-processing cost.

