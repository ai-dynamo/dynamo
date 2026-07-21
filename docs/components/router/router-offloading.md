---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Offloading Support Matrix
subtitle: Per-framework support and setup for KV routing with tiered KV cache offloading
---

This page is the cross-framework support matrix for running the Dynamo KV router together with KV cache offloading: which cache tiers the router can see on each framework, what to configure, and which versions you need. Setup details live in the per-framework sections below and the guides they link.

Every combination starts the frontend the same way:

```bash
python -m dynamo.frontend --http-port 8000 --router-mode kv
```

The worker-side setup differs per framework, and so does what the router can see. Two support levels matter:

- **Router-visible (tier-aware).** The worker publishes KV cache events annotated with the storage tier (`medium`). The router tracks per-tier residency and credits lower-tier prefix hits when selecting workers, weighted by `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight`.
- **Engine-side only.** Offloading works inside each worker, but events carry no usable tier information — the router routes on GPU-tier cache state and treats offloaded prefixes as cache misses.

> [!NOTE]
> Offloading support changes quickly on both the framework and the Dynamo side. Version gates are stated in the matrix cells where they limit support, with full version requirements in the per-framework sections below; capabilities merged upstream but not yet in a release are marked not yet supported. vLLM tier-aware routing and the `--router-host-cache-hit-weight` / `--router-disk-cache-hit-weight` tuning flags require Dynamo 1.3.0 or later; SGLang HiCache tier-aware routing also works on Dynamo 1.2.x, with the lower-tier weights fixed at their defaults.

## Support Matrix

Legend: ✅ tier-aware routing · 🟡 engine-side offloading only (router routes on GPU-tier state) · ❌ not yet supported · — does not exist for this framework.

| Framework | GPU | CPU RAM | Disk | Shared pool |
| --- | --- | --- | --- | --- |
| [**SGLang**](#sglang) | ✅ KV events | ✅ HiCache + KV events — SGLang 0.5.11+ | — no separate disk tier; HiCache's third tier is the shared pool (next column) | ✅ HiCache + Mooncake + `--shared-cache-type hicache` |
| [**TensorRT-LLM**](#tensorrt-llm) | ✅ `--publish-kv-events` | 🟡 native host cache (`host_cache_size`) is not router-visible | — no native disk tier | — |
| [**vLLM**](#vllm) | ✅ KV events | ✅ `OffloadingConnector` + self-describing KV events — vLLM v0.24.0+, aggregated serving | ❌ multi-tier events are not in a released vLLM | ❌ no router integration for vLLM shared-pool connectors |

[KVBM](../kvbm/README.md), [LMCache](../../integrations/lmcache-integration.md), and [FlexKV](../../integrations/flexkv-integration.md) add CPU and disk tiers outside the frameworks' native paths; their router interaction is covered in [Other Offloading Backends](#other-offloading-backends).

## SGLang

Enable KV event publishing on every worker, plus HiCache for the host tier:

- Worker: `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'` and the SGLang-native HiCache flags (`--enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through`).
- For the Mooncake shared pool: add `--hicache-storage-backend mooncake` (plus its extra config) on the workers and `--shared-cache-type hicache --shared-cache-multiplier 0.5` on the frontend.
- Versions: SGLang 0.5.11 or later emits host-tier (`CPU_PINNED`) events; earlier versions offload engine-side but the router only sees the GPU tier. With Mooncake, use SGLang 0.5.13 or later to avoid a bundled-Mooncake crash.

See [Using HiCache](../../backends/sglang/sglang-hicache.md) for full setup, scoring, verification, and troubleshooting.

## TensorRT-LLM

Enable KV event publishing on every worker:

- Worker: `--publish-kv-events` (Dynamo 1.3.0+; earlier releases and current examples use the `--publish-events-and-metrics` spelling, which remains accepted as a deprecated alias and logs a deprecation warning). Requires the PyTorch backend.
- TensorRT-LLM events carry no tier annotation, so only the GPU tier is router-visible.
- Native host offloading (`kv_cache_config.host_cache_size` and `secondary_offload_min_priority`, passed through `--extra-engine-args`) runs under the KV router and extends each worker's reuse capacity, but the router cannot see the host tier.
- For CPU and disk tiers managed by Dynamo, use KVBM (`--connector kvbm`; see [Other Offloading Backends](#other-offloading-backends)).

See the [TensorRT-LLM backend docs](../../backends/trtllm/README.md) for worker setup.

## vLLM

Enable KV event publishing and native CPU offloading with self-describing events on every worker:

- Worker: `--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'` and `--kv-transfer-config` with `"kv_connector": "OffloadingConnector"` and `"self_describing_kv_events": true` in `kv_connector_extra_config`.
- Versions: vLLM v0.24.0 or later. Earlier versions publish placeholder CPU events that the router silently drops — offloading still works engine-side, but the router only sees the GPU tier.
- Disk and multi-tier offloading (`TieringOffloadingSpec`): router-usable events are not yet available in a released vLLM.

See [Native KV Offloading](../../backends/vllm/vllm-native-kv-offloading.md) for the full support matrix (including disaggregated and tensor-parallel status), setup commands, verification, and troubleshooting.

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
| `--router-disk-cache-hit-weight` | `0.25` | Disk-tier prefix overlap (no framework produces this tier today) |
| `--shared-cache-type` / `--shared-cache-multiplier` | `none` / `0.5` | **Experimental.** Shared-pool lookups (SGLang HiCache + Mooncake only) |

See [Configuration and Tuning](router-configuration.md) for the cache-hit weight semantics, [Using HiCache](../../backends/sglang/sglang-hicache.md#configuration) for the shared-cache flags, and [Router Operations](router-operations.md) for enabling event publishing per backend.

## See Also

- [Router Guide](router-guide.md) — routing modes, deployment modes, and event transport
- [Native KV Offloading](../../backends/vllm/vllm-native-kv-offloading.md) — vLLM detail page
- [Using HiCache](../../backends/sglang/sglang-hicache.md) — SGLang detail page
- [KV Cache Offloading](../../backends/vllm/vllm-kv-offloading.md) — vLLM offloading backend catalog (native, KVBM, LMCache, FlexKV)
- [Feature Matrix](../../reference/feature-matrix.md) — per-backend feature support across Dynamo
