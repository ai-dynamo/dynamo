---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Offloading Support Matrix
subtitle: Per-framework support and setup for KV routing with tiered KV cache offloading
---

Use this matrix to see which KV offloading tiers the Dynamo KV router can use for each framework.

## Support Matrix

Legend: ✅ tier-aware routing · 🟡 router-visible, tier-agnostic · 🚧 Dynamo integration in progress · ❌ not yet supported · — does not exist for this framework.

| Framework | Version gates | GPU | CPU RAM | Disk | Shared pool |
| --- | --- | --- | --- | --- | --- |
| [**vLLM**](#vllm) | vLLM v0.24.0+; Dynamo v1.3.0+ | ✅ KV events | ✅ `OffloadingConnector` + self-describing KV events (aggregated) | 🚧 vLLM main emits FS/OBJ events; Dynamo tier mapping is in progress | 🚧 vLLM locality events are merged; Dynamo shared-pool indexing is in progress |
| [**SGLang**](#sglang) | SGLang v0.5.11+; v0.5.13+ with Mooncake; Dynamo v1.2+ | ✅ KV events | ✅ HiCache + KV events | — no separate disk tier; HiCache's third tier is the shared pool (next column) | ✅ HiCache + Mooncake + `--shared-cache-type hicache` |
| [**TensorRT-LLM**](#tensorrt-llm) | Dynamo v1.3.0+ for the current event flag | 🟡 `--publish-kv-events`; merged GPU + RAM view | 🟡 native host cache shares one router view with GPU; per-tier weights do not apply | — no native disk tier | — |

[KVBM](../kvbm/README.md), [LMCache](../../integrations/lmcache-integration.md), and [FlexKV](../../integrations/flexkv-integration.md) add CPU and disk tiers outside the frameworks' native paths; their router interaction is covered in [Other Offloading Backends](#other-offloading-backends).

### Status Definitions

- **Router-visible (tier-aware).** The worker publishes KV cache events annotated with the storage tier (`medium`). The router tracks per-tier residency and credits lower-tier prefix hits when selecting workers, weighted by `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight`.
- **Router-visible (tier-agnostic).** The router keeps blocks indexed across engine-managed tiers but cannot distinguish which tier currently holds them, so per-tier weights do not apply.

> [!NOTE]
> Offloading support changes quickly on both the framework and the Dynamo side. Version gates are summarized in the matrix and expanded in the per-framework sections below. Capabilities merged upstream but not yet released are listed as main-branch support. vLLM tier-aware routing and the `--router-host-cache-hit-weight` / `--router-disk-cache-hit-weight` tuning flags require Dynamo 1.3.0 or later; SGLang HiCache tier-aware routing also works on Dynamo 1.2.x, with the lower-tier weights fixed at their defaults.

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

Enable KV event publishing and native CPU offloading with self-describing events on every worker:

- Worker: `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'` and `--kv-transfer-config` with `"kv_connector": "OffloadingConnector"` and `"self_describing_kv_events": true` in `kv_connector_extra_config`.
- Versions: vLLM v0.24.0 or later. Earlier versions publish placeholder CPU events that the router silently drops — offloading still works engine-side, but the router only sees the GPU tier.
- Disk and multi-tier offloading (`TieringOffloadingSpec`): vLLM main emits FS and OBJ events. Dynamo tier mapping is in progress.
- Shared pools: vLLM main publishes optional `LOCAL` / `REMOTE` locality metadata on FS and OBJ events. Dynamo shared-pool indexing is in progress.

See [Native KV Offloading](../../backends/vllm/vllm-native-kv-offloading.md) for the full support matrix (including disaggregated and tensor-parallel status), setup commands, verification, and troubleshooting.

## SGLang

Enable KV event publishing on every worker, plus HiCache for the host tier:

- Worker: `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'` and the SGLang-native HiCache flags (`--enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through`).
- For the Mooncake shared pool: add `--hicache-storage-backend mooncake` (plus its extra config) on the workers and `--shared-cache-type hicache --shared-cache-multiplier 0.5` on the frontend.
- Versions: SGLang 0.5.11 or later emits host-tier (`CPU_PINNED`) events; earlier versions offload engine-side but the router only sees the GPU tier. With Mooncake, use SGLang 0.5.13 or later to avoid a bundled-Mooncake crash.

See [Using HiCache](../../backends/sglang/sglang-hicache.md) for full setup, scoring, verification, and troubleshooting.

## TensorRT-LLM

Enable KV event publishing on every worker:

- Worker: `--publish-kv-events` (Dynamo 1.3.0+; earlier releases and current examples use the `--publish-events-and-metrics` spelling, which remains accepted as a deprecated alias and logs a deprecation warning). Requires the PyTorch backend.
- TensorRT-LLM events do not identify GPU versus host RAM. The router keeps a block indexed while it remains in either tier and removes it after it leaves the lowest tier.
- Native host offloading (`kv_cache_config.host_cache_size` and `secondary_offload_min_priority`, passed through `--extra-engine-args`) therefore extends router-visible reuse, but host-tier weights do not apply.
- For CPU and disk tiers managed by Dynamo, use KVBM (`--connector kvbm`; see [Other Offloading Backends](#other-offloading-backends)).

See the [TensorRT-LLM backend docs](../../backends/trtllm/README.md) for worker setup.

## Other Offloading Backends

These backends offload KV cache outside the frameworks' native paths. KVBM and FlexKV ship launch scripts that run under `--router-mode kv`; LMCache's shipped examples pair it with the router only in the disaggregated script. In every case, the column to check is what the router sees.

| Backend | Frameworks | Tiers | Router interaction |
| --- | --- | --- | --- |
| [KVBM](../kvbm/README.md) | vLLM, TensorRT-LLM | CPU, CPU + disk | Offloaded blocks stay in the router's index as a merged device-tier view: a block evicted from GPU but held in KVBM CPU or disk still earns routing credit. Per-tier weights do not apply. |
| [LMCache](../../integrations/lmcache-integration.md) | vLLM | CPU, local/remote storage | No documented router integration. The router routes on GPU-tier events; LMCache tiers are worker-internal. |
| [FlexKV](../../integrations/flexkv-integration.md) | vLLM | CPU, SSD | Runs with KV routing; the router routes on GPU-tier events, and FlexKV tiers are worker-internal. |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | SGLang (HiCache) | Shared external pool | Tier-aware and shared-pool-aware as the SGLang HiCache storage backend (the SGLang row above). Dynamo has no direct vLLM + Mooncake integration; with vLLM, Mooncake appears only as an optional LMCache storage backend. |

Launch scripts that combine these backends with the KV router: [`agg_kvbm_router.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_kvbm_router.sh), [`disagg_kvbm_router.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/disagg_kvbm_router.sh), and [`agg_flexkv_router.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_flexkv_router.sh).

## Router Flags for Lower Tiers

The router-side knobs are backend-independent; set them on the frontend when workers publish tier-annotated events:

| Flag | Default | Applies to |
| --- | --- | --- |
| `--router-host-cache-hit-weight` | `0.75` | CPU-tier prefix overlap (SGLang HiCache, vLLM `OffloadingConnector`) |
| `--router-disk-cache-hit-weight` | `0.25` | Disk-tier prefix overlap when the backend publishes a Dynamo-recognized disk tier |
| `--shared-cache-type` / `--shared-cache-multiplier` | `none` / `0.5` | **Experimental.** Shared-pool lookups (SGLang HiCache + Mooncake only) |

See [Configuration and Tuning](router-configuration.md) for the cache-hit weight semantics, [Using HiCache](../../backends/sglang/sglang-hicache.md#configuration) for the shared-cache flags, and [Router Operations](router-operations.md) for enabling event publishing per backend.

## See Also

- [Router Guide](router-guide.md) — routing modes, deployment modes, and event transport
- [Native KV Offloading](../../backends/vllm/vllm-native-kv-offloading.md) — vLLM detail page
- [Using HiCache](../../backends/sglang/sglang-hicache.md) — SGLang detail page
- [KV Cache Offloading](../../backends/vllm/vllm-kv-offloading.md) — vLLM offloading backend catalog (native, KVBM, LMCache, FlexKV)
- [Feature Matrix](../../reference/feature-matrix.md) — per-backend feature support across Dynamo
