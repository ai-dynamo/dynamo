---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Offloading Support Matrix
subtitle: Per-framework support and setup for KV routing with tiered KV cache offloading
---

Use this matrix to see which KV offloading tiers the Dynamo KV router can use for each framework.

## Support Matrix

Legend: ✅ tier-aware routing · ⚠️ available but not fully validated · 🟡 router-visible, tier-agnostic · 🚧 Dynamo integration in progress · ❌ not yet supported · — does not exist for this framework.

| Framework | Version gates | GPU | CPU RAM | Disk | Shared pool |
| --- | --- | --- | --- | --- | --- |
| [**vLLM**](#vllm) | vLLM v0.24.0+; Dynamo v1.3.0+ | ✅ KV events | ✅ `OffloadingConnector` + self-describing KV events (aggregated) | 🚧 vLLM main emits FS and OBJ events; Dynamo tier mapping is in progress | 🚧 vLLM main emits optional locality metadata; Dynamo shared-pool indexing is in progress |
| [**SGLang**](#sglang) | SGLang v0.5.11+; v0.5.13+ with Mooncake; Dynamo v1.2+ | ✅ KV events | ✅ HiCache + KV events | — no separate disk tier; HiCache's third tier is the shared pool (next column) | ✅ HiCache + Mooncake + `--shared-cache-type hicache` |
| [**TensorRT-LLM**](#tensorrt-llm) | Dynamo v1.3.0+ for the current event flag | 🟡 `--publish-kv-events`; merged GPU + RAM view | 🟡 native host cache shares one router view with GPU; per-tier weights do not apply | — no native disk tier | — |

### Status Definitions

- **Router-visible (tier-aware).** The worker publishes KV cache events annotated with the storage tier (`medium`). The router tracks per-tier residency and credits lower-tier prefix hits when selecting workers, weighted by `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight`.
- **Router-visible (tier-agnostic).** The router keeps blocks indexed across engine-managed tiers but cannot distinguish which tier currently holds them, so per-tier weights do not apply.

> [!NOTE]
> Offloading support changes quickly. Version gates are summarized in the matrix and expanded in the per-framework sections below.

## Common Frontend Setup

Every combination starts the frontend the same way:

```bash
python -m dynamo.frontend --http-port 8000 --router-mode kv
```

### Verify Router Visibility

After sending requests, confirm that the router is applying KV events:

```bash
curl -s localhost:8000/metrics | grep kv_cache_events_applied
```

A rising `event_type="stored",status="ok"` counter confirms event ingestion,
but not which tiers are visible. For tier-aware backends, run the frontend with
`DYN_LOG=debug` and look for `Queried lower-tier indexer` messages such as
`storage_tier=HostPinned`. TensorRT-LLM uses a merged GPU + RAM view, so it has
no separate CPU-tier signal. The router does not currently compare a worker's
offloading configuration with the tiers it observes.

## vLLM

Start each vLLM worker with native CPU offloading and router-usable KV events:

```bash
PYTHONHASHSEED=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --block-size 16 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"block_size":256,"self_describing_kv_events":true}}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

- Versions: vLLM v0.24.0 or later. Earlier versions publish placeholder CPU events that the router silently drops — offloading still works engine-side, but the router only sees the GPU tier.
- Disk and multi-tier offloading (`TieringOffloadingSpec`): vLLM main emits FS and OBJ events; Dynamo tier mapping is in progress.
- Shared pools: vLLM main emits optional locality metadata; Dynamo shared-pool indexing is in progress.

See [Native KV Offloading](../backends/vllm/native-kv-offloading.md) for the full support matrix (including disaggregated and tensor-parallel status), setup commands, verification, and troubleshooting.

## SGLang

Start each SGLang worker with HiCache and KV event publishing:

```bash
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --skip-tokenizer-init
```

- For the Mooncake shared pool: add `--hicache-storage-backend mooncake` (plus its extra config) on the workers and `--shared-cache-type hicache --shared-cache-multiplier 0.5` on the frontend.
- Versions: SGLang 0.5.11 or later emits host-tier (`CPU_PINNED`) events; earlier versions offload engine-side but the router only sees the GPU tier. With Mooncake, use SGLang 0.5.13 or later to avoid a bundled-Mooncake crash.

See [Using HiCache](../backends/sglang/hicache.md) for full setup, scoring, verification, and troubleshooting.

## TensorRT-LLM

Start each TensorRT-LLM worker with native host offloading and KV event publishing:

```bash
python3 -m dynamo.trtllm \
  --model Qwen/Qwen3-0.6B \
  --publish-kv-events \
  --trtllm.kv_cache_config.enable_block_reuse true \
  --trtllm.kv_cache_config.host_cache_size 17179869184 \
  --trtllm.kv_cache_config.secondary_offload_min_priority 35
```

- Versions: `--publish-kv-events` requires Dynamo 1.3.0 or later and the PyTorch backend. Earlier releases use the deprecated `--publish-events-and-metrics` alias.
- TensorRT-LLM events do not identify GPU versus host RAM. The router keeps a block indexed while it remains in either tier and removes it after it leaves the lowest tier.
- Native host offloading therefore extends router-visible reuse, but host-tier weights do not apply.

See the [TensorRT-LLM backend docs](../backends/tensorrt-llm/overview.md) for worker setup.

## Router Flags for Lower Tiers

### How the router actually computes

The Rust worker selector (`lib/kv-router/src/scheduling/selector.rs`) scores each candidate worker with:

```
overlap_credit_blocks = device_overlap_blocks       * effective_device_credit
                      + host_pinned_overlap_blocks  * 0.75  // host_cache_hit_weight, configurable in router config
                      + disk_overlap_blocks         * 0.25  // disk_cache_hit_weight, configurable in router config
                      + shared_overlap_blocks       * shared_cache_multiplier

adjusted_prefill_blocks = max(raw_prefill_blocks - overlap_credit_blocks, 0)

logit = prefill_load_scale * adjusted_prefill_blocks   // default 1.0
      + decode_blocks                                    // active/projected KV load
      + decode_active_request_weight * active_requests  // optional, default 0
```

Before applying the device term, the router computes
`effective_device_credit = overlap_score_credit / (1 + overlap_score_credit_decay * normalized_excess_prefill)`.
The host, disk, and shared-cache credits do not decay.

The router picks the worker with the **minimum** logit.

### Router Flags

These flags only affect routing. Values range from 0 to 1: 0 ignores lower-tier KV cache hits, while 1 treats them like GPU cache hits.

| Flag | Default | Effect |
| --- | --- | --- |
| `--router-host-cache-hit-weight` | `0.75` | Credit for each matched host-pinned block |
| `--router-disk-cache-hit-weight` | `0.25` | Credit for each matched disk or external-tier block |
| `--shared-cache-type` / `--shared-cache-multiplier` | `none` / `0.5` | **Experimental.** Enable shared-cache lookup and credit matches beyond the device-local prefix |

See [Configuration and Tuning](configuration-and-tuning.md) for the cache-hit weight semantics, [Using HiCache](../backends/sglang/hicache.md#configuration) for the shared-cache flags, and [Router Operations](router-operations.md) for enabling event publishing per backend.

## See Also

- [Router Guide](router-guide.md) — routing modes, deployment modes, and event transport
- [Native KV Offloading](../backends/vllm/native-kv-offloading.md) — vLLM detail page
- [Using HiCache](../backends/sglang/hicache.md) — SGLang detail page
- [KV Cache Offloading](../backends/vllm/kv-cache-offloading.md) — vLLM offloading backend catalog (native, KVBM, LMCache, FlexKV)
- [Feature Matrix](../../../../reference/general/compatibility.mdx) — per-backend feature support across Dynamo
