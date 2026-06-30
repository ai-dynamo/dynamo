---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Route multimodal requests to workers with the best KV cache overlap
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image content when computing cache overlap scores. An image hash (`mm_hash`) is computed per request — in the Rust frontend by default for vLLM backends, by vLLM's own processor when the chat-processor variant is enabled, or by a dedicated MM router worker for TRT-LLM backends — and included in per-block routing metadata. The KV router then selects the backend worker with the highest cache overlap, including overlap on image embedding blocks.

Repeated requests containing the same image are routed to the worker that already has the corresponding KV cache blocks, maximizing prefix cache reuse.

> Note: KV cache is separate from embedding cache (also called encoder cache), which reuses vision encoder outputs (image→embeddings) to avoid re-running the encoder. For encoder-side reuse see [Embedding Cache](embedding-cache.md).

## When to Use

Use multimodal KV routing when:

- You have multiple backend workers serving multimodal requests
- Your workload includes repeated images across requests (e.g., the same product photo, shared reference images)
- You want to maximize KV cache hit rates for multimodal content

Without MM-aware routing, the standard router treats image token blocks as opaque and cannot match which worker has cached a particular image's KV blocks.

## Support Matrix

| Backend | Path | Supported | Notes |
|---------|------|-----------|-------|
| **vLLM** | Rust frontend (default) | ✅ | Uses Dynamo-owned image-token counting and placeholder expansion for the model families listed below. |
| **vLLM** | Python chat-processor (`--dyn-chat-processor vllm --router-mode kv`) | ✅ | Uses vLLM's own multimodal processor — supports any VLM that vLLM supports. |
| **TRT-LLM** | — | ✅ | Uses dedicated MM Router Worker. Requires `--publish-events-and-metrics` on TRT-LLM workers. |
| **SGLang** | Rust frontend (default) | ✅ (\*) | Uses the same Dynamo-owned image-token counters; engaged automatically when the worker reports `backend_framework="sglang"`. |

(\*) The SGLang Rust-frontend path substitutes per-image `pad_value` tokens in the routing-side view so SGLang's RadixAttention prefix cache key (`MM_PAD_SHIFT_VALUE + mm_hash % 2^30`) is identical to the frontend key. Requires the SGLang fork with the `mm_hashes` field on `GenerateReqInput` ([sgl-project/sglang#25300](https://github.com/sgl-project/sglang/pull/25300)).

## Supported Model Families (Rust frontend path)

The Rust frontend owns six image-token counting algorithms and recognizes ten
model families. It matches the model's `model_type` first, then its model ID.

| Counter | Model Families | Routing Status |
|---------|----------------|----------------|
| Qwen2 | Qwen2-VL, Qwen2.5-VL | Enabled |
| Qwen3 | Qwen3-VL, Qwen3.5, Qwen3.6 | Enabled |
| LLaVA-1.5 | LLaVA-1.5 | Enabled |
| LLaVA-NeXT | LLaVA-NeXT | Enabled |
| Llama 4 | Llama 4 | Compatibility behavior; see warning below |
| Kimi K2 | Kimi-K2.5, Kimi-K2.6 | Enabled |

A model outside this list falls back to text-prefix-only KV routing. The
request still completes, but routing cannot account for cache overlap across
images.

> [!WARNING]
> The Llama 4 counter and single-placeholder expansion preserve Dynamo's current
> behavior. vLLM applies pixel shuffle to produce 144 image positions per tile
> and adds structural tokens, so the Rust routing sequence differs from vLLM's
> processed sequence. Use the Python chat-processor path when routing must follow
> vLLM's structured Llama 4 sequence.

The Python chat-processor variant doesn't share this constraint — it
delegates to vLLM's own multimodal processor and works with any VLM vLLM
supports.

## How It Works

### vLLM (default — Rust frontend)

```text
Frontend (Rust + KV router) → Backend Workers
        │
        ├─ Hash image (xxh3_64 of the raw URL — full-URL identity; use --frontend-decoding for content-addressed hashing)
        ├─ Resolve image-token id from model config or the Dynamo tokenizer
        ├─ Read (W, H) from a Range: 0-65535 header fetch (or in-memory data: bytes)
        ├─ count_tokens(W, H) → expanded image-token count N
        ├─ Expand placeholder × N in routing_token_ids (worker token_ids unchanged)
        ├─ Encode mm_hash in per-image pad-value tokens
        ├─ KV router selects best worker
        └─ Forward mm_hash to worker via extra_args["mm_hashes"] →
              vLLM's multi_modal_uuids (cache key match)
```

1. With frontend decoding, the Rust frontend computes `mm_hash` as `xxh3_64` over the decoded tensor identity: rank and shape, data type, and RGB payload. Otherwise it hashes the full URL string, including a complete `data:` URI. Passthrough reuse therefore requires identical URLs, while decoded mode reuses identical decoded tensors across URLs.
2. The frontend selects one of the ten model families and resolves its image-placeholder token ID. It reads the applicable `config.json` field (`image_token_id`, `image_token_index`, or `media_placeholder_token_id`) and falls back to exact Dynamo tokenizer vocabulary lookup when the model registers only a placeholder string. Llama 4 retains the compatibility behavior described above.
3. Per-image `(W, H)` is read from a 64KB `Range`-bounded header fetch (or from in-memory bytes for `data:` URIs); the selected Dynamo-owned counter computes the expanded token count.
4. The single placeholder token is expanded to N copies in `routing_token_ids` (a router-only view); the worker still sees one placeholder per image in `token_ids`.
5. Each replacement token encodes the image's `mm_hash` as a canonical `pad_value`. `block_mm_infos` remains empty because the image identity already resides in the routing token stream.
6. The frontend forwards each image's `mm_hash` as 16 hexadecimal characters via `extra_args["mm_hashes"]`; the backend pads it to vLLM's 64-character `multi_modal_uuids` form, so vLLM's own KV-cache key matches the hash the router used.

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

### SGLang

```text
Frontend (Rust + KV router) → SGLang Workers
        │                                                       │
        ├─ Hash image (same xxh3_64 path as vLLM Rust)           │
        ├─ Resolve image-token id + (W, H) (same as vLLM Rust)   │
        ├─ count_tokens(W, H) → expanded count N                 │
        ├─ Expand placeholder × N in routing_token_ids           │
        ├─ KV router selects best worker                         │
        ├─ Substitute pad_value per image in worker token_ids:   │
        │     pad_value = MM_PAD_SHIFT_VALUE + (mm_hash % 2^30)  │
        └─ Forward mm_hash list via GenerateReqInput.mm_hashes ──┘
                                                                  │
                                          SGLang server seeds     │
                                          MultimodalDataItem.hash │
                                          from mm_hashes;         │
                                          set_pad_value() honors  │
                                          the preset hash → its   │
                                          internal pad_value      │
                                          matches the router's    │
                                          → RadixAttention key    │
                                          alignment.              │
```

Unlike the vLLM path (which forwards `mm_hashes` as `multi_modal_uuids` for vLLM's own KV-event publisher to consume), SGLang's RadixAttention computes its cache key from the **token IDs** of the prompt, including the per-image `pad_value` token that gets inserted in place of image placeholders. The router has to substitute that `pad_value` itself in its token-id view so its overlap calculation matches what the worker will actually cache.

Two preconditions produce identical routing-side and server-side hashes:

1. **Dynamo Rust frontend** computes `pad_value = MM_PAD_SHIFT_VALUE + (mm_hash % 2^30)` for each image and substitutes that value (× N expansion) in `routing_token_ids`. Engaged automatically when the worker's `ModelDeploymentCard` reports `backend_framework="sglang"`.
2. **SGLang fork** exposes `GenerateReqInput.mm_hashes: Optional[List[str]]`. When set, `set_pad_value()` skips its internal `hash_feature()` recompute and uses the caller's hash directly, so the worker's derived `pad_value` matches the router's substitution. See [upstream PR sgl-project/sglang#25300](https://github.com/sgl-project/sglang/pull/25300).

Without both pieces, the routing-side hash and the server-side hash decouple and MM-aware routing silently degrades to text-prefix-only (the request still completes; just no prefix-cache benefit across images).

## Launching

### vLLM (default — Rust frontend)

```bash
cd $DYNAMO_HOME
bash examples/backends/vllm/launch/agg_multimodal_router.sh
```

The Rust frontend computes per-image token counts with Dynamo-owned counters
and expands placeholders in-process, so the router can match the backend's
expanded image-token count without invoking the Hugging Face image processor. Each
`mm_hash` is then forwarded to the worker as `multi_modal_uuids` so vLLM's
KV events publish the same key the router computes.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Model to serve |
| `NUM_WORKERS` | `2` | Number of backend workers |
| `BLOCK_SIZE` | `16` | KV cache block size (must match backend) |
| `GPU_MEMORY_UTILIZATION` | `0.20` | Per-worker GPU memory fraction |
| `SINGLE_GPU` | `false` | Pack all workers onto GPU 0 (testing-only override; pass `--single-gpu` or set `SINGLE_GPU=true` for functional tests on a single-GPU box) |
| `KV_EVENTS_PORT_BASE` | `5557` | Worker `i` publishes ZMQ KV events on `BASE + i - 1` |
| `DYN_LOG` | `info,mm_routing=debug,...` | Frontend log filter |
| `VLLM_EXTRA_ARGS` | (unset) | Pass-through args to `python -m dynamo.vllm`. Set `--frontend-decoding` to enable content-addressed `mm_hash` (cross-URL KV-cache reuse). |

To opt into frontend image decoding (so the frontend downloads + decodes once and `mm_hash` becomes content-addressed instead of URL-addressed):

```bash
VLLM_EXTRA_ARGS="--frontend-decoding" \
    bash examples/backends/vllm/launch/agg_multimodal_router.sh
```

The worker then registers a `media_decoder` on its model card; the frontend's `MediaLoader` runs in-process and hashes the decoded tensor's shape, data type, and RGB payload via xxh3. Two distinct signed URLs that decode to the same tensor use the same routing key.

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

### SGLang

```bash
cd $DYNAMO_HOME
bash examples/backends/sglang/launch/agg_multimodal_router.sh
```

The launcher sets `--kv-events-config` on each worker plus the standard `--router-mode kv --kv-cache-block-size 16` on the frontend so the KV router consumes block-level overlap. The SGLang pad-value protocol is engaged automatically — the worker registers `backend_framework="sglang"` in its `ModelDeploymentCard` and the frontend picks the matching path from there.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Model to serve |
| `NUM_WORKERS` | `2` | Number of backend SGLang workers |
| `BLOCK_SIZE` | `16` | SGLang `--page-size`; must match Frontend `--kv-cache-block-size` |
| `SINGLE_GPU` | `false` | Pack all workers onto GPU 0 (for single-GPU functional tests) |
| `KV_EVENTS_PORT_BASE` | `29090` | Worker `i` publishes ZMQ KV events on `BASE + i - 1` |
| `DYN_LOG` | `info,mm_routing=debug,…` | Frontend log filter |
| `SGLANG_EXTRA_ARGS` | (unset) | Pass-through args to `python -m dynamo.sglang` |

Both prerequisites are enabled by default in the dynamo sglang image; the list below is only for users building dynamo from source:

- Dynamo built with `--features mm-routing` (Rust frontend's image-token counter + the SGLang glue). The container build always passes this; the cargo feature is opt-in only for source builds.
- SGLang with the `GenerateReqInput.mm_hashes` field — added by [sgl-project/sglang#25300](https://github.com/sgl-project/sglang/pull/25300) and vendored into the dynamo sglang image via `container/deps/sglang/patches/<ver>/`.

  If you're running against your own sglang install (outside the dynamo image), apply it manually:

  ```bash
  SITE_PACKAGES_ROOT="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent.parent)')"
  cd "$SITE_PACKAGES_ROOT"
  curl -sL https://github.com/sgl-project/sglang/pull/25300.diff | python3 -c '
  import sys
  chunks = sys.stdin.read().split("diff --git ")
  filtered = [c for c in chunks if c.startswith("a/python/sglang/")]
  print("".join("diff --git " + c for c in filtered), end="")
  ' > /tmp/sglang_pr25300_python_only.diff
  patch --dry-run -p2 < /tmp/sglang_pr25300_python_only.diff
  patch -p2 < /tmp/sglang_pr25300_python_only.diff
  cd -
  ```

  Without the patch, the worker's `_resolve_mm_hashes_supported` probe
  falls through to text-prefix-only routing (no crash, but no MM-aware
  cache reuse either).

Verification (visible at `DYN_LOG=info,mm_routing=debug`):

```text
mm_routing: MM-aware KV routing enabled              model=Qwen/Qwen3-VL-2B-Instruct  …
mm_routing: image-token count                        tokens=1024  mm_hash=17828397777369824042  …
mm_routing: MmRoutingInfo built (exact)              n_images=4  block_size=16  total_tokens=8416  n_blocks=526
dynamo_llm::kv_router::push_router: [ROUTING] Best: worker_X with N/M blocks overlap
```

On the second identical request the same worker wins with high overlap (e.g. `137/138 blocks overlap` on Qwen3-VL-2B with 4 images), confirming MM-aware reuse.

## Transfer Mode Details (vLLM chat-processor variant only)

Applies to the `--dyn-chat-processor=vllm` launch (`agg_multimodal_router_chat_processor.sh`), **not** the default Rust frontend path. In the chat-processor variant the frontend runs the HF image processor in-process and ships the pre-processed `mm_kwargs` to the selected backend worker so the backend can skip re-processing; the `DYNAMO_MM_TRANSFER` environment variable controls how that payload is transferred.

The default Rust frontend path doesn't run the HF processor or pre-render `mm_kwargs` — it forwards only `mm_hashes`, and each worker re-processes the image itself. TRT-LLM backends similarly re-run their own preprocessing and don't honor `DYNAMO_MM_TRANSFER`.

- **`shm`** (default): POSIX shared memory via a `/dev/shm` segment. Intended for same-node deployments, where frontend and backend share the host filesystem. If the backend can't access the segment (e.g., running on a different node), it falls back to re-processing the image from the URL.
- **`nixl`**: NIXL RDMA transfer. Required for cross-node deployments where `/dev/shm` is not shared between frontend and backend. Works across nodes over InfiniBand or TCP (whichever UCX selects).
- **`DYNAMO_DISABLE_NIXL_MM=1`**: Disables pre-processed mm_kwargs transfer entirely. The backend downloads and processes images itself from the original URLs. Useful for debugging or when transfer overhead exceeds re-processing cost.
